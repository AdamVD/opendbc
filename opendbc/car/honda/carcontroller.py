import math
import threading
from queue import Empty, Queue

import numpy as np
from openpilot.common.params import Params

from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, rate_limit, make_tester_present_msg, structs
from opendbc.car.honda import hondacan
from opendbc.car.honda.values import CAR, CruiseButtons, HondaFlags, HONDA_BOSCH, HONDA_BOSCH_CANFD, HONDA_BOSCH_RADARLESS, \
                                     HONDA_BOSCH_TJA_CONTROL, HONDA_NIDEC_ALT_PCM_ACCEL, CarControllerParams
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.common.pid import PIDController

from opendbc.sunnypilot.car.honda.mads import MadsCarController
from opendbc.sunnypilot.car.honda.gas_interceptor import GasInterceptorCarController
from opendbc.sunnypilot.car.honda.icbm import IntelligentCruiseButtonManagementInterface

VisualAlert = structs.CarControl.HUDControl.VisualAlert
LongCtrlState = structs.CarControl.Actuators.LongControlState


def compute_gb_honda_bosch(accel, speed):
  # TODO returns 0s, is unused
  return 0.0, 0.0


def compute_gb_honda_nidec(accel, speed):
  creep_brake = 0.0
  creep_speed = 2.3
  creep_brake_value = 0.15
  if speed < creep_speed:
    creep_brake = (creep_speed - speed) / creep_speed * creep_brake_value
  gb = float(accel) / 4.8 - creep_brake
  return np.clip(gb, 0.0, 1.0), np.clip(-gb, 0.0, 1.0)


def compute_gas_brake(accel, speed, fingerprint):
  if fingerprint in HONDA_BOSCH:
    return compute_gb_honda_bosch(accel, speed)
  else:
    return compute_gb_honda_nidec(accel, speed)


# TODO not clear this does anything useful
def actuator_hysteresis(brake, braking, brake_steady, v_ego, car_fingerprint):
  # hyst params
  brake_hyst_on = 0.02    # to activate brakes exceed this value
  brake_hyst_off = 0.005  # to deactivate brakes below this value
  brake_hyst_gap = 0.01   # don't change brake command for small oscillations within this value

  # *** hysteresis logic to avoid brake blinking. go above 0.1 to trigger
  if (brake < brake_hyst_on and not braking) or brake < brake_hyst_off:
    brake = 0.
  braking = brake > 0.

  # for small brake oscillations within brake_hyst_gap, don't change the brake command
  if brake == 0.:
    brake_steady = 0.
  elif brake > brake_steady + brake_hyst_gap:
    brake_steady = brake - brake_hyst_gap
  elif brake < brake_steady - brake_hyst_gap:
    brake_steady = brake + brake_hyst_gap
  brake = brake_steady

  return brake, braking, brake_steady


def brake_pump_hysteresis(apply_brake, apply_brake_last, last_pump_ts, ts):
  pump_on = False

  # reset pump timer if:
  # - there is an increment in brake request
  # - we are applying steady state brakes and we haven't been running the pump
  #   for more than 20s (to prevent pressure bleeding)
  if apply_brake > apply_brake_last or (ts - last_pump_ts > 20. and apply_brake > 0):
    last_pump_ts = ts

  # once the pump is on, run it for at least 0.2s
  if ts - last_pump_ts < 0.2 and apply_brake > 0:
    pump_on = True

  return pump_on, last_pump_ts


def process_hud_alert(hud_alert):
  alert_fcw = False
  alert_steer_required = False

  # Make sure FCW is prioritized over steering required
  # TODO: implement separate available LDW alert
  if hud_alert == VisualAlert.fcw:
    alert_fcw = True
  elif hud_alert in (VisualAlert.steerRequired, VisualAlert.ldw):
    alert_steer_required = True

  return alert_fcw, alert_steer_required


class HondaParamWriter:
  def __init__(self):
    self._params = Params()
    self._queue = Queue()
    self._thread = threading.Thread(target=self._run, name="honda-param-writer", daemon=True)
    self._thread.start()

  def put_many(self, values):
    self._queue.put({key: float(value) for key, value in values.items()})

  def _run(self):
    while True:
      pending = self._queue.get()

      # Collapse queued snapshots so delayed writes keep only the newest value per key.
      try:
        while True:
          pending.update(self._queue.get_nowait())
      except Empty:
        pass

      for key, value in pending.items():
        self._params.put_nonblocking(key, value)


class CarController(CarControllerBase, MadsCarController, GasInterceptorCarController, IntelligentCruiseButtonManagementInterface):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    MadsCarController.__init__(self)
    GasInterceptorCarController.__init__(self, CP, CP_SP)
    IntelligentCruiseButtonManagementInterface.__init__(self, CP, CP_SP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.params = CarControllerParams(CP)
    self.CAN = hondacan.CanBus(CP)
    self.tja_control = CP.carFingerprint in HONDA_BOSCH_TJA_CONTROL
    self.param_writer = HondaParamWriter()

    self.braking = False
    self.brake_steady = 0.
    self.brake_last = 0.
    self.apply_brake_last = 0
    self.last_pump_ts = 0.
    self.stopping_counter = 0

    self.accel = 0.0
    self.speed = 0.0
    self.gas = 0.0
    self.brake = 0.0
    self.last_torque = 0.0
    self.bosch_last_gas = 0

    self.last_pcm_off = 0.0  # for rate-limiting in model-based NIDEC FF
    self.brake_hold_frames = 0    # POST-gate committed-bite countdown (runs in the 50 Hz block)
    self.brake_hold_ramp = 0.0    # rate-limited FELT-FLOOR the hold carries apply_brake up to
    self.brake_hold_pos_frames = 0  # debounce for the "plan wants to go" early release
    self.dither_lfsr = 0x5A  # servo-ID dither PRBS-7 state (NIDEC_DITHER_* in values.py)
    self.a_des_lp = 0.0      # servo-aware FF: low-pass of a_des for the build-vs-ease latch
    self.po_build = False    # servo-aware FF: True while the accel request is building (onset)
    self.dbo_sus = 0         # DBO: frames the demand has exceeded the nudge-cap reach (sustain gate)
    self.knee_guard = True   # DBO knee guard: True while demand is sub-threshold (guarded by default)
    self.lift_guard = False  # low-gear lift guard: engine-rpm latch (NIDEC_LIFT_GUARD_* in values.py)
    self.gas_override_linger = 0  # frames to hold the override path through the release handback

    # Tier-1 kickdown-surge trim state (see NIDEC_TRIM_* in values.py)
    self.trim_frames = 0          # frames remaining in the trim/recovery window
    self.last_tgt_gear = 0        # last valid TRANS_TARGET_GEAR (0 = unknown)
    self.po_filt = 0.0            # ~1s low-pass of pcm_off (the TCU schedule sees history)

    # Descent-mode latch state (NIDEC_DESCENT_* in values.py, SPEC_descent_mode_2026-07-11)
    self.descent = False         # engine-brake-first latch
    self.descent_bte = False     # band-top exited: re-entry below BAND_REARM or after REARM_T cooldown
    self.descent_bte_frames = 0  # frames since band-top exit (drives the REARM_T cooldown, rev 3)
    self.descent_pitch_lp = 0.0  # ~1s LP of pitch for the latch only (FF paths keep raw pitch)
    self.descent_capable = self.params.NIDEC_DESCENT and CP.carFingerprint in HONDA_NIDEC_ALT_PCM_ACCEL

    self.gasfactor = 1.0 if (Params().get("HondaGasFactorParams") is None) else Params().get("HondaGasFactorParams")
    self.gasfactor_before_maxgas = self.gasfactor
    self.windfactor = 1.0 if (Params().get("HondaWindFactorParams") is None) else Params().get("HondaWindFactorParams")
    self.windfactor_before_maxgas = self.windfactor_before_brake = self.windfactor
    self.pitch = 0.0

    # Bosch extra-brake controller
    self.brake_pid = PIDController(k_p=([0,], [0,]),
                                   k_i=([0.], [0.5]),
                                   pos_limit=0.0,
                                   neg_limit=-2.0,
                                   rate=50)
    self.brake_pid.reset()

  def update(self, CC, CC_SP, CS, now_nanos):
    MadsCarController.update(self, self.CP, CC, CC_SP)
    gas_pedal_force = 0.0
    actuators = CC.actuators
    hud_control = CC.hudControl
    # NIDEC_ALT friction demand, hoisted so the 50 Hz brake block below can read it on frames
    # where the NIDEC_ALT longActive branch did not run. The only other reads are inside that
    # branch (the knee bias and the brake fraction), so this is a pure hoist -- no revert gate.
    friction_ms2 = 0.0
    hud_v_cruise = hud_control.setSpeed / CS.v_cruise_factor if hud_control.speedVisible else 255
    pcm_cancel_cmd = CC.cruiseControl.cancel

    if len(CC.orientationNED) == 3:
      self.pitch = CC.orientationNED[1]
    hill_brake = math.sin(self.pitch) * ACCELERATION_DUE_TO_GRAVITY
    # For NIDEC_ALT (Odyssey): Honda ACC self-rejects most grade, so the effective grade
    # leakage at the FF operating point is ~2.2 m/s^2, not 9.81. Use this for BOTH the
    # pcm_off gas path and the direct brake path so they stay mutually exclusive and
    # neither over-compensates for grade (see NIDEC_MODEL_GRADE_G in values.py).
    hill_brake_ff = math.sin(self.pitch) * self.params.NIDEC_MODEL_GRADE_G
    # ~1s LP of pitch for the descent-mode latch only (runs every frame on descent-capable
    # cars so the filter is already settled when the latch conditions are first evaluated)
    if self.descent_capable:
      self.descent_pitch_lp += (DT_CTRL / self.params.NIDEC_DESCENT_PITCH_TAU) * (self.pitch - self.descent_pitch_lp)

    # Low-gear lift guard latch (NIDEC_LIFT_GUARD, FINDINGS_uphill_follow_2026-07-18): engine rpm
    # is the kickdown-state proxy -- above RPM_ON the throttle-shut lift plant is bimodal (torque
    # persists, then fuel-cut snaps well past mild asks), so decel asks must stay in the servo
    # modulation band and let friction serve the remainder. Hysteretic; rpm=0 (no signal) -> off.
    engine_rpm = getattr(CS, "engine_rpm", 0.0)
    if self.params.NIDEC_LIFT_GUARD_RATIO:
      # 7/19 latch-leak fix (FINDINGS_overshoot_pull_2026-07-19): rpm hysteresis kept the guard
      # on through normal mid-gear cruise (9th ~1660 rpm @ 50mph > RPM_OFF) -- 36.4% of engaged
      # time. The gear ratio rpm/kph separates cruise (<=21.1) from retained-kickdown (26.9+)
      # cleanly; below XSPD_MIN the ratio is slip noise and the guard stays off (PCM long is
      # floor-canceled below 21.5 mph anyway).
      xspd = getattr(CS, "xmission_speed", 0.0)
      gear_ratio = engine_rpm / xspd if xspd > self.params.NIDEC_LIFT_RATIO_XSPD_MIN else 0.0
      if gear_ratio > self.params.NIDEC_LIFT_RATIO_ON:
        self.lift_guard = True
      elif gear_ratio < self.params.NIDEC_LIFT_RATIO_OFF:
        self.lift_guard = False
    elif engine_rpm > self.params.NIDEC_LIFT_GUARD_RPM_ON:
      self.lift_guard = True
    elif engine_rpm < self.params.NIDEC_LIFT_GUARD_RPM_OFF:
      self.lift_guard = False
    # descent-mode (prev frame's latch/trim state) owns its envelope -- guard stands aside there
    lift_guard_active = (self.params.NIDEC_LIFT_GUARD and self.lift_guard
                         and not self.descent and not self.descent_bte)

    # Gas-override handoff (Odyssey): keep the PCM speed-servo commanded while the driver
    # presses the accelerator. The PCM arbitrates pedal-vs-servo max-wins (stock Honda
    # pedal-override semantics), so the servo rides under the foot and the takeover on
    # release has no torque gap -- without this it rebuilds from coast over ~2.5-3s, a
    # -0.5 m/s^2 sag (FINDINGS_override_handoff_2026-06-11). Friction brake stays zero
    # while the pedal is down (the gas/brake block below keeps brake=0 when not
    # longActive, and panda blocks 0x1FA on gas press regardless).
    gas_override_long = (CC.enabled and not CC.longActive
                         and CS.out.gasPressed and not CS.out.brakePressed
                         and self.CP.carFingerprint in HONDA_NIDEC_ALT_PCM_ACCEL
                         and not self.CP_SP.enableGasInterceptor)
    # Release handback bridge: carState's gasPressed falls one cycle before controlsd's
    # longActive comes back, so on EVERY pedal release there are stale frames where
    # neither longActive nor gas_override_long holds -> the not-long branch below resets
    # the pcm_off slew state, and if a 10Hz ACC_HUD tick lands in the window (~1 in 5
    # releases) a literal PCM_SPEED=0 reaches the PCM, dumping the servo to engine-brake
    # and rebuilding over ~3s -- the exact sag this path exists to kill (route 35 ep2,
    # 2026-06-11: a_min -0.65 vs -0.07/+0.10 on the clean releases). Linger ~0.3s after
    # the pedal falls while still enabled (brake or disengage clears immediately).
    if gas_override_long:
      self.gas_override_linger = int(0.3 / DT_CTRL)
    elif not CC.enabled or CS.out.brakePressed or CC.longActive:
      self.gas_override_linger = 0
    elif self.gas_override_linger > 0:
      self.gas_override_linger -= 1
      gas_override_long = True

    if CC.longActive:
      accel = actuators.accel
      if (self.CP.carFingerprint in (CAR.ACURA_MDX_3G, CAR.ACURA_MDX_3G_MMR)) and (accel > max(0, CS.out.aEgo) + 0.1):
        accel = 10000.0 # help with lagged accel until pedal tuning is inserted
      if self.CP.carFingerprint in HONDA_NIDEC_ALT_PCM_ACCEL:
        # Two-plant fix (FINDINGS_channel_plants_2026-06-09): the PCM servo self-rejects grade ONLY
        # while its throttle is active (g_eff ~2.2, NIDEC_MODEL_GRADE_G). As the request goes to decel
        # the throttle closes and gravity acts in full (9.81) — uphill mild-decel must lift LESS
        # (gravity already brakes; was the uphill over-decel/lead-oscillation), downhill braking must
        # brake MORE (was the under-brake + integrator-windup droop). Blend continuously on the decel
        # request to avoid chatter. The pcm_off path below uses the SAME blended hill term, preserving
        # gas-vs-brake mutual exclusivity (friction starts exactly where PCM authority ends).
        w_passive = float(np.clip((-actuators.accel - self.params.NIDEC_MODEL_GRADE_BLEND_LO) /
                                  (self.params.NIDEC_MODEL_GRADE_BLEND_HI - self.params.NIDEC_MODEL_GRADE_BLEND_LO),
                                  0.0, 1.0))
        g_blend = self.params.NIDEC_MODEL_GRADE_G + (ACCELERATION_DUE_TO_GRAVITY - self.params.NIDEC_MODEL_GRADE_G) * w_passive
        hill_brake_ff = math.sin(self.pitch) * g_blend
        # 2026-07-25: the grade credit may never push the COMMAND positive on a decel ask.
        # pcm_off derives from a_des = actuators.accel + hill_brake_ff (below), so uphill the
        # credit flipped a decel ask into a gas request: over 115 engaged min, mild-decel frames
        # (accel in [-0.6,-0.1]) with pcm_off > 0 ran 3.2% at 1-2% grade, 18.2% at 2-3%, 49.4% at
        # 3-4% and 75.6% above +4%. That BOTH commanded the PCM above vEgo against the ask AND
        # grade-activated the `last_pcm_off > 0` friction guardrail below, which then zeroed the
        # brake exactly where the ask needed serving. With this clamp the >+4% figure is 1.1%.
        # Clamp to EXACTLY 0.0, not -eps: `a_des >= 0` branch membership is then unchanged, so
        # servo-aware ease, DBO and the knee-guard latch stay bit-identical (verified over 115
        # engaged min: kguard/dbo_sus differ on 0.000% of frames). The friction demand is unchanged
        # by construction (a_des_b > 0 and a_des_b == 0 both give fric_demand = 0) -- also verified
        # bit-identical, so this is a PURE command-side change.
        # ⚠ deliberately inside `if CC.longActive:` -- during gas_override_long the po path uses the
        # unblended sin(pitch)*GRADE_G above and must keep the servo under the driver's foot.
        if self.params.NIDEC_GRADE_CREDIT_CLAMP and actuators.accel < 0.0:
          hill_brake_ff = min(hill_brake_ff, -actuators.accel)
        # Honest Plant-B friction demand: cover decel beyond engine-brake + TRUE aero coastdown, scaled
        # by the MEASURED brake plant (3.07 m/s^2 at full apply_brake, not compute_gas_brake's 4.8).
        # Replaces both the /4.8 map and the downstream wind_brake subtraction (which together
        # under-commanded the brake ~2x on flat ground; droop deficit decomposition +0.95 m/s^2).
        wind_ms2 = float(np.interp(CS.out.vEgo, [0.0, 13.4, 22.4, 31.3, 40.2], [0.000, 0.049, 0.136, 0.267, 0.441]))
        a_des_b = actuators.accel + hill_brake_ff
        # Demand-ramped engine-brake credit (2026-07-11, FINDINGS_gov_first_drive): the flat 0.05
        # is the PASSIVE-coast figure (6/10 measurement, TC unlocked); under a commanded pcm_off
        # lift the PCM keeps the TC locked and holds/downshifts gear, so the lift channel alone
        # delivers ~EB(v) at the floor while the DECEL_SOFT map is ALREADY commanding it to serve
        # the full ask -- crediting only 0.05 made friction re-serve the same demand (stacked on
        # 64% of small decel holds, delivered x1.91; factory friction duty ~4%). Ramp the credit
        # with demand depth (po floors near |a_des| ~ 0.4 via the DECEL_SOFT map).
        eb_credit = self.params.NIDEC_MODEL_ENGINE_BRAKE
        if self.params.NIDEC_MODEL_EB_DYN:
          eb_full = float(np.interp(CS.out.vEgo, self.params.NIDEC_MODEL_EB_BP, self.params.NIDEC_MODEL_EB_V))
          eb_credit = float(np.interp(-a_des_b, [0.0, self.params.NIDEC_MODEL_EB_FULL_AT],
                                      [self.params.NIDEC_MODEL_ENGINE_BRAKE, eb_full]))
        if lift_guard_active:
          # kickdown state: the EB credit is unschedulable (bimodal lift plant) -- let the
          # measured-linear friction channel serve what gravity cannot (uphill blend already
          # priced gravity into a_des_b). Pairs with the LIFT_PO_FLOOR clamp on the pcm_off path.
          eb_credit = 0.0
        # Descent-mode latch (SPEC_descent_mode_2026-07-11, design rev 2 after the 7/12 review):
        # engine-brake-first descents. All gates hysteretic; v_cruise = HUD set speed (m/s,
        # driver cluster set; guarded by speedVisible -- a stale 255 also fails the band by
        # construction). The wire to the planner's DESCENT_* tolerance floor is actuators.accel
        # (the raw PID output) -- NOT a_des_b: the grade-FF blend multiplies small PID dips by
        # up to ~2.4x on steep grades and self-released the latch exactly where it matters
        # (review 7/12). While the planner floor tracks delivered accel, actuators.accel rides
        # ~0; a real released demand (lead, curve, SCC/SLA, e2e) drives it past ADES_MIN via
        # the feedforward within a frame or two. With a VISIBLE LEAD any sustained intent past
        # LEAD_ADES releases -- mild lead asks must never be tolerated. Entry additionally
        # requires the current friction demand ~ 0 (seamless engage, no one-frame brake dump).
        fric_demand = max(0.0, -a_des_b - eb_credit - wind_ms2)
        # Honest friction while guarded-and-clamped (2026-07-19, FINDINGS_overshoot_pull): the
        # condition mirrors the po clamp below -- on these frames the command is pinned in the
        # hold band, so the PCM servo holds speed with the throttle OPEN. Aero (wind_ms2) and
        # the gravity blend (inside a_des_b) are credits for a LIFTED throttle and are phantom
        # here; subtracting them created a light-decel dead zone (asks to -0.5 flat / -0.9 on
        # 6% up delivered ~0) that ended in one oversized brake at short range (d4 t1038:
        # cb 1.85 at THW 1.6). Serve the raw ask with friction alone -- the servo cancels grade
        # and aero symmetrically (hold band holds aEgo~0 on any grade, plant ID Q3).
        if lift_guard_active and self.params.NIDEC_LIFT_HONEST_FRICTION and \
           (actuators.accel + hill_brake_ff < 0.0 or actuators.accel < 0.0):
          fric_demand = max(0.0, -actuators.accel)
        # P6: brake too small to feel should not be used at all -- it is pad wear, a VSA pump cycle
        # and a brake-light flicker for zero authority (0 of Adam's own 117 applications deliver
        # < 0.20 m/s^2). This is ALSO the coast-authority rule: measured closed-throttle authority
        # in gears 7-10 at 20-40 m/s is 0.13-0.25 m/s^2 grade-free (corpus, CAR_GAS <= 4 counts;
        # the identified plant agrees at 0.176 in 10th @32 m/s) -- "too small to feel" and "smaller
        # than coasting delivers anyway" are the same threshold on this car.
        # POSITION IS LOAD-BEARING: before the descent block, so the floored fric_demand is what
        # reaches `friction_ms2` below and therefore what gates the committed bite (a demand
        # dropped here can no longer commit a dwell).
        # NOTE this also feeds the descent-entry test below (fric_demand < DESCENT_FRIC_ENTRY);
        # measured effect on descent coverage: +0.00 pp (quartet), +0.07 pp (trio).
        # fric_demand is >= 0 by construction, so MIN_AMP = 0.0 is an exact revert.
        if fric_demand < self.params.NIDEC_FRIC_MIN_AMP:
          fric_demand = 0.0
        if self.params.NIDEC_DESCENT and not gas_override_long:
          v_err = CS.out.vEgo - hud_control.setSpeed  # >0 = over set (m/s)
          if self.descent and v_err >= self.params.NIDEC_DESCENT_BAND_TOP:
            self.descent_bte = True  # band-top exit: friction trims; re-arm below REARM or on cooldown
            self.descent_bte_frames = 0
          elif self.descent_bte:
            # rev 3 (first drive 7/12): steep-grade friction equilibrium (+0.5..+1.0) sits ABOVE the
            # REARM line, so geometry alone locked the latch out for the rest of the hill after one
            # excursion (a8: 17.8 s of friction). Time-bound the sawtooth instead: instant re-arm when
            # the grade eases (below REARM, as before) OR after REARM_T seconds of trim.
            self.descent_bte_frames += 1
            if v_err < self.params.NIDEC_DESCENT_BAND_REARM or \
               self.descent_bte_frames >= int(self.params.NIDEC_DESCENT_REARM_T / DT_CTRL):
              self.descent_bte = False
          valid = hud_control.speedVisible and CS.out.vEgo > self.params.NIDEC_DESCENT_V_MIN
          pitch_ok = self.descent_pitch_lp < (self.params.NIDEC_DESCENT_PITCH_OFF if self.descent
                                              else self.params.NIDEC_DESCENT_PITCH_ON)
          # First entry anywhere in the band (steep-descent friction equilibrium sits ~+2 kph,
          # between REARM and TOP -- descent_check.py 370/975 frames); REARM only bounds the
          # post-band-top trim sawtooth.
          band_hi = self.params.NIDEC_DESCENT_BAND_REARM if (not self.descent and self.descent_bte) \
                    else self.params.NIDEC_DESCENT_BAND_TOP
          band_ok = self.params.NIDEC_DESCENT_BAND_LOW < v_err < band_hi
          ades_ok = actuators.accel > (self.params.NIDEC_DESCENT_ADES_MIN if self.descent
                                       else self.params.NIDEC_DESCENT_ADES_REARM)
          lead_ok = not (hud_control.leadVisible and actuators.accel < self.params.NIDEC_DESCENT_LEAD_ADES)
          entry_ok = self.descent or fric_demand < self.params.NIDEC_DESCENT_FRIC_ENTRY
          self.descent = valid and pitch_ok and band_ok and ades_ok and lead_ok and entry_ok
        else:
          self.descent = False
          self.descent_bte = False
        # In-latch the band IS the demand server (PCM engine brake via its own downshift);
        # past-band / released frames are byte-identical to baseline.
        friction_ms2 = 0.0 if self.descent else fric_demand
        creep_brake = ((2.3 - CS.out.vEgo) / 2.3 * 0.15) if CS.out.vEgo < 2.3 else 0.0  # legacy stop-hold
        # knee bias: the first ~13 counts produce no decel (hydraulic preload), so demanded friction
        # rides on top of the knee -- keeps light braking (descents, gentle stops) from landing in
        # the dead zone
        knee = self.params.NIDEC_MODEL_BRAKE_KNEE if friction_ms2 > 0.0 else 0.0
        brake = float(np.clip(friction_ms2 / self.params.NIDEC_MODEL_BRAKE_PLANT + knee + creep_brake, 0.0, 1.0))
        gas = float(np.clip(actuators.accel / 4.8, 0.0, 1.0))  # legacy gas frac (unused on this path)
      else:
        gas, brake = compute_gas_brake(actuators.accel + hill_brake, CS.out.vEgo, self.CP.carFingerprint)
    else:
      accel = 0.0
      gas, brake = 0.0, 0.0
      self.descent = False
      self.descent_bte = False

    # *** rate limit steer ***
    limited_torque = rate_limit(actuators.torque, self.last_torque, -self.params.STEER_DELTA_DOWN * DT_CTRL,
                                self.params.STEER_DELTA_UP * DT_CTRL)
    self.last_torque = limited_torque

    # *** apply brake hysteresis ***
    pre_limit_brake, self.braking, self.brake_steady = actuator_hysteresis(brake, self.braking, self.brake_steady,
                                                                           CS.out.vEgo, self.CP.carFingerprint)

    # *** rate limit after the enable check ***
    self.brake_last = rate_limit(pre_limit_brake, self.brake_last, -2., 3 * DT_CTRL)

    # vehicle hud display, wait for one update from 10Hz 0x304 msg
    alert_fcw, alert_steer_required = process_hud_alert(hud_control.visualAlert)

    # **** process the car messages ****

    # steer torque is converted back to CAN reference (positive when steering right)
    apply_torque = int(np.interp(-limited_torque * self.params.STEER_MAX,
                                 self.params.STEER_LOOKUP_BP, self.params.STEER_LOOKUP_V))

    speed_control = 1 if ((accel <= 0.0) and (CS.out.vEgo == 0)) else 0

    # Send CAN commands
    can_sends = []

    # tester present - w/ no response (keeps radar disabled)
    if self.CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS) and self.CP.openpilotLongitudinalControl:
      if self.frame % 10 == 0:
        can_sends.append(make_tester_present_msg(0x18DAB0F1, self.CAN.pt, suppress_response=True))

    # Send steering command.
    can_sends.append(hondacan.create_steering_control(self.packer, self.CAN, apply_torque, CC.latActive, self.tja_control))

    # wind brake from air resistance decel at high speed
    wind_brake = np.interp(CS.out.vEgo, [0.0, 2.3, 35.0], [0.001, 0.002, 0.15]) * self.windfactor # not in m/s2 units
    wind_brake_ms2 = np.interp(CS.out.vEgo, [0.0, 13.4, 22.4, 31.3, 40.2], [0.000, 0.049, 0.136, 0.267, 0.441]) # in m/s2 units

    # all of this is only relevant for HONDA NIDEC
    speed_control = 0
    max_accel = np.interp(CS.out.vEgo, self.params.NIDEC_MAX_ACCEL_BP, self.params.NIDEC_MAX_ACCEL_V)
    # TODO this 1.44 is just to maintain previous behavior
    pcm_speed_BP = [-wind_brake,
                    -wind_brake * (3 / 4),
                    0.0,
                    0.5]
    # The Honda ODYSSEY seems to have different PCM_ACCEL
    # msgs, is it other cars too?
    if self.CP_SP.enableGasInterceptor or not (CC.longActive or gas_override_long):
      pcm_speed = 0.0
      pcm_accel = int(0.0)
      self.last_pcm_off = 0.0
      self.a_des_lp = 0.0
      self.po_build = False
      self.dbo_sus = 0
      self.knee_guard = True
      self.trim_frames = 0
      self.last_tgt_gear = 0
      self.po_filt = 0.0
    elif self.CP.carFingerprint in HONDA_NIDEC_ALT_PCM_ACCEL:
      # Model-based feedforward: invert the identified plant aego = K(v)*pcm_off - g*sin(pitch),
      # so pcm_off = (accel + hill_brake) / K(v), clamped to [PCM_OFF_MIN, PCM_OFF_MAX].
      # ASYMMETRIC gain: accel side uses the full 1/K (strong throttle authority = the "oomph");
      # decel side uses 1/(K*DECEL_SOFT). Honda's ACC throttle only modulates over pcm_off [0,-1.5]
      # (coast/engine-brake, saturating ~-0.3 m/s2; it does NOT friction-brake), so the symmetric
      # high gain slammed the throttle shut for any small ease-off — a relay/cliff that feeds the
      # lead-follow oscillation. The softer decel gain grades the lift-off. pcm_off stays <=0 on any
      # decel (a_des<0 -> pcm_off<0), so the ACC never pushes gas against the direct brake; hard
      # decel still reaches the -1.5 floor where the brake channel handles it.
      K_v = float(np.clip(self.params.NIDEC_MODEL_K0 - self.params.NIDEC_MODEL_K1 * CS.out.vEgo,
                          self.params.NIDEC_MODEL_K_MIN, self.params.NIDEC_MODEL_K_MAX))
      a_des = actuators.accel + hill_brake_ff  # hill_brake_ff shared with direct-brake path (computed above)
      # effective inverse-gain: full K on accel, K*DECEL_SOFT on decel (gentler, graded lift-off)
      K_eff = K_v * (self.params.NIDEC_MODEL_DECEL_SOFT if a_des < 0.0 else 1.0)
      pcm_off = float(np.clip(a_des / K_eff, self.params.NIDEC_MODEL_PCM_OFF_MIN, self.params.NIDEC_MODEL_PCM_OFF_MAX))

      # Servo/saturation-aware ease (NIDEC_MODEL_SERVO_* in values.py). The map above inverts a linear
      # K; the Honda PCM is a saturating speed servo, so while EASING we instead invert its real
      # proportional gain Ks=accel_ceil/PO_EDGE (the command lands in the responsive band) and park at
      # the band edge while saturated -> the servo eases the instant the plan drops below the ceiling.
      # Build-vs-ease is the exogenous a_des trend (hysteresis latch, no chatter). Onset (po_build)
      # keeps the full 1/K offset above; decel (a_des<0) keeps the DECEL_SOFT path. SERVO_AWARE=False
      # leaves the baseline map untouched. Skipped under gas_override_long: the override-handoff keeps
      # pcm_speed primed for a smooth release, and the driver's pedal dominates anyway (PCM max-wins).
      if self.params.NIDEC_MODEL_SERVO_AWARE:
        self.a_des_lp += (DT_CTRL / self.params.NIDEC_MODEL_SERVO_TREND_TAU) * (a_des - self.a_des_lp)
        if a_des - self.a_des_lp > self.params.NIDEC_MODEL_SERVO_TREND_EPS:
          self.po_build = True
        elif a_des - self.a_des_lp < -self.params.NIDEC_MODEL_SERVO_TREND_EPS:
          self.po_build = False
        if a_des >= 0.0 and not self.po_build and not gas_override_long:
          accel_ceil = float(np.clip(self.params.NIDEC_MODEL_CEIL_V0 - self.params.NIDEC_MODEL_CEIL_K * CS.out.vEgo,
                                     self.params.NIDEC_MODEL_CEIL_MIN, self.params.NIDEC_MODEL_CEIL_MAX))
          if a_des >= accel_ceil:
            pcm_off = self.params.NIDEC_MODEL_PO_EDGE + self.params.NIDEC_MODEL_PO_HOLD_MARGIN  # saturated: park at band edge
          else:
            pcm_off = a_des / (accel_ceil / self.params.NIDEC_MODEL_PO_EDGE)                    # in-band: invert servo gain
          pcm_off = float(np.clip(pcm_off, self.params.NIDEC_MODEL_PCM_OFF_MIN, self.params.NIDEC_MODEL_PCM_OFF_MAX))

      # DBO -- onset front-load + recoverability CAP (NIDEC_MODEL_DBO / NIDEC_DBO_* in values.py).
      # Applied ON TOP of the servo-aware FF above (does NOT replace it). The pcm_off knee IS a
      # downshift and within a gear pcm_off ~does nothing (0x130 plant-ID), so:
      #  (1) "INDUCE TORQUE FASTER" (FRONT_LOAD, default OFF since 2026-07-11) -- on a genuine RISING
      #      demand the baseline parks pcm_off in its ease band (~1.5, BELOW the ~2.5 knee) for a
      #      gradual gap-close => it SAGS until the gap forces a fast ramp; front-load to KNEE_CROSS
      #      fires the mild downshift promptly. DISABLED because the deliberate knee-cross is exactly
      #      the "0.3 held gets 0.6" late-overshoot mechanism (sustained-hold 2026-07-10) and chases
      #      the opening gap factory deliberately does not (follow-policy 2026-07-09) -- see values.py.
      #  (2) "NEVER UNRECOVERABLE OVER-ACCEL" -- baseline rails pcm_off to ~5-8 (INERT: aego identical
      #      4..8) and oozes back over 1-2s = the confirmed over-pull/overshoot (Jekyll Check-2). Cap
      #      at PO_CAP bounds the pull (gear-7 at pcm_off 3.2 ~0.35 vs railed ~0.6) and retracts in
      #      ~one 0.85s lag -> tames the deep-kickdown overshoot (it does NOT prevent the downshift
      #      DEPTH itself -- that needs a follow gear-hold, deferred). Uncaps to FULL_CAP once the
      #      demand SUSTAINS > SUSTAIN_T AND asks > UNCAP_A (2026-07-10 fix: the time gate alone was
      #      tripped by 53% of patient follow holds -- the a_des gate keeps set-speed bumps /
      #      pull-aways / sustained uphill at full authority while a 0.2-0.45 follow ask stays capped).
      # DAMP (experimental, default 0) coasts sub-floor asks. Decel & DBO=False are byte-identical to
      # the servo-aware baseline; the else-reset avoids a stale sustain-uncap on the gas-override
      # handback (safety review 2026-06-19). Front-load uses po_build (maintained by the servo-aware
      # block above); with SERVO_AWARE off it degrades to cap-only.
      if self.params.NIDEC_MODEL_DBO and a_des >= 0.0 and not gas_override_long:
        if self.params.NIDEC_DBO_DAMP > 0.0 and a_des < self.params.NIDEC_DBO_DAMP:
          pcm_off = 0.0
        elif self.params.NIDEC_DBO_FRONT_LOAD and self.po_build and a_des > self.params.NIDEC_DBO_ONSET_MIN:
          pcm_off = max(pcm_off, self.params.NIDEC_DBO_KNEE_CROSS)  # onset front-load (induce torque faster)
        # Sustain-uncap gate (reviewed 2026-07-11): the a_des threshold tracks the planner's own
        # speed-dependent accel ceiling (a railed ask IS full demand at any speed -- a fixed 0.5 was
        # unreachable above ~36 m/s where the planner caps asks at 0.54..0.50), and the counter gets
        # hysteresis + a freeze band so unfiltered-pitch dither in a_des around the threshold can
        # neither zero the 1.3s counter nor flap an earned uncap back to PO_CAP mid pull-away.
        uncap_a = float(np.interp(CS.out.vEgo, self.params.NIDEC_DBO_UNCAP_A_BP, self.params.NIDEC_DBO_UNCAP_A_V))
        # Knee guard (2026-07-11 first-drive, FINDINGS_gov_first_drive): sub-threshold demand parks
        # at the servo band edge (2.2) so the command CANNOT cross the ~2.5 downshift knee -- small
        # asks (incl. every governor-capped 0.35 chase ask) mapped to pcm_off 2-3 and delivered
        # x1.9-2.4 through the kickdown, defeating the cap. Latch shares the uncap gate's hysteresis
        # band: ON below (uncap_a - HYST), OFF above uncap_a, held in between (pitch dither can't
        # flap it). Genuine demand (a_des > uncap_a, grade FF included) is untouched: same-frame
        # release to the PO_CAP/FULL_CAP ladder below.
        if self.params.NIDEC_DBO_KNEE_GUARD > 0.0:
          # KNEE_GUARD_RAW (2026-07-18): latch on the RAW ask, not grade-FF-inflated a_des --
          # on steep grades the FF term alone released the guard for every capped chase ask,
          # firing the kickdown at the worst moment (uphill-follow limit cycle). A real climb
          # still releases: speed sag drives the raw PID ask past uncap_a within a few seconds.
          guard_a = actuators.accel if self.params.NIDEC_DBO_KNEE_GUARD_RAW else a_des
          if guard_a < uncap_a - self.params.NIDEC_DBO_UNCAP_A_HYST:
            self.knee_guard = True
          elif guard_a > uncap_a:
            self.knee_guard = False
          # Lead hold (2026-07-19, FINDINGS_overshoot_pull): the raw release still fired a
          # commanded kickdown INTO the follow mark (d3 t529: uphill speed-sag released the
          # guard with a matched lead at THW 2.2 -> 10 s of +0.3 pull against a 0-capped plan,
          # overshoot to THW 0.90). With a lead visible, hold the 2.2 park regardless of the
          # release state -- the PCM self-downshifts under sustained load at the park when it
          # truly must (observed d4 t999; stock never commands kickdown depth either). Also
          # floors the sustain-uncap ladder below (pcm_off <= 2.2 < PO_CAP never counts up).
          # No lead = exact 7/18 release.
          if self.knee_guard or (self.params.NIDEC_DBO_LEAD_HOLD and hud_control.leadVisible):
            pcm_off = min(pcm_off, self.params.NIDEC_DBO_KNEE_GUARD)
        if pcm_off > self.params.NIDEC_DBO_PO_CAP and a_des > uncap_a:
          self.dbo_sus += 1
        elif pcm_off <= self.params.NIDEC_DBO_PO_CAP or a_des < uncap_a - self.params.NIDEC_DBO_UNCAP_A_HYST:
          self.dbo_sus = 0
        # else: a_des inside the hysteresis band with demand still above the cap -> hold the counter
        cap = (self.params.NIDEC_DBO_FULL_CAP if self.dbo_sus * DT_CTRL >= self.params.NIDEC_DBO_SUSTAIN_T
               else self.params.NIDEC_DBO_PO_CAP)
        pcm_off = min(pcm_off, cap)
      elif self.params.NIDEC_MODEL_DBO:
        self.dbo_sus = 0  # DBO on but not capping (decel/override): reset so no stale uncap on handback

      # Tier-1 kickdown-surge trim: on the TCU's downshift announcement (TRANS_TARGET_GEAR
      # down-step) in a power-on context, scale the ask by TRIM_DEPTH and recover over
      # TRIM_DECAY_S. A chained announcement (multi-gear kickdown, ~25% of events) restarts
      # the clock at the same depth. The upstream long PID will lean against the trim for
      # its 3s life; that is intended (the surge it offsets is larger).
      tg = getattr(CS, "trans_target_gear", 0)
      if 1 <= tg <= 10:
        if CC.longActive and tg < self.last_tgt_gear <= self.params.NIDEC_TRIM_GFROM_MAX and \
           self.po_filt > self.params.NIDEC_TRIM_PO_MIN and \
           actuators.accel > -0.3 and not CS.out.brakePressed:
          self.trim_frames = int(self.params.NIDEC_TRIM_DECAY_S / DT_CTRL)
        self.last_tgt_gear = tg
      if self.trim_frames > 0:
        if pcm_off > 0.0:
          prog = 1.0 - self.trim_frames * DT_CTRL / self.params.NIDEC_TRIM_DECAY_S
          pcm_off *= self.params.NIDEC_TRIM_DEPTH + (1.0 - self.params.NIDEC_TRIM_DEPTH) * prog
        self.trim_frames -= 1

      # Low-gear lift guard clamp (NIDEC_LIFT_GUARD, FINDINGS_uphill_follow_2026-07-18): while
      # the engine is in kickdown state and the RAW ask is a decel, pin the command inside the
      # measured hold band -- pcm_off in [-0.3, +0.3] delivers aEgo ~0 on any grade (plant ID Q3,
      # 97k frames), while anything outside it in low gear either keeps pulling (+0.15 sustained
      # for any po>0.3) or snaps through fuel-cut EB to ~2x the ask. Two-sided ON PURPOSE and
      # keyed on actuators.accel: on steep grades the w_passive gravity blend INVERTS a deep
      # decel ask into positive a_des (route c2 t=54s: act_a -0.37 -> wire po +3.2), so the
      # accel-side paths above (band-edge park, DBO ladder) treat it as gas demand -- the raw
      # sign is the true intent. Friction (eb_credit zeroed above) serves what gravity cannot.
      # Placed after all accel-path shaping; the descent anchor below still overrides in-latch
      # (lift_guard_active already excludes descent/bte frames).
      if lift_guard_active and (a_des < 0.0 or actuators.accel < 0.0):
        pcm_off = float(np.clip(pcm_off, self.params.NIDEC_LIFT_PO_FLOOR, self.params.NIDEC_LIFT_PO_CEIL))

      # Descent-mode anchor (SPEC_descent_mode_2026-07-11, rev 2): while latched, present the
      # PCM the stock descent signal -- PCM_SPEED anchored near set while vEgo grows = growing
      # overspeed error (stock's own downshift trigger). The bias pre-load is PROPORTIONAL to
      # overspeed (up to BIAS at v_err >= BIAS): the downshift fires ~2 kph earlier than stock
      # once real overspeed develops, while the anchored equilibrium stays AT set -- a constant
      # pre-load parked it 2 kph under set on grades the servo can hold (review 7/12). Clip
      # [PCM_OFF_MIN, -BAND_LOW]: never below the envelope we already command daily; the small
      # positive headroom (+0.5) lets the servo gas back to set on drag-dominant gentle grades
      # (stock behavior: partial throttle holds set) instead of sagging into the BAND_LOW exit.
      # The existing slew below rate-limits transitions both directions.
      if self.descent:
        v_err_anchor = CS.out.vEgo - hud_control.setSpeed
        bias_eff = min(self.params.NIDEC_DESCENT_BIAS, max(0.0, v_err_anchor))
        pcm_off = float(np.clip((hud_control.setSpeed - bias_eff) - CS.out.vEgo,
                                self.params.NIDEC_MODEL_PCM_OFF_MIN, -self.params.NIDEC_DESCENT_BAND_LOW))

      # Servo-ID dither (2026-07-24, plant-simulator campaign): PRBS-7 overlay inside the
      # measured null band (pcm_off in [-0.3,+0.3] holds aEgo~0 on any grade -- uphill
      # plant-ID, 97k frames). Persistent excitation that passive driving never provides:
      # breaks the closed-loop po<->state correlation and identifies servo tau + small-signal
      # s[gear] (the map cells three independent fits disagree on 2x). INERT by default
      # (AMP=0.0); when armed, applies only in calm cruise -- no lead, no override, tiny ask,
      # not in descent/lift-guard, v>15 m/s. Perceptual bound: 0.3 * K~0.12 ~= 0.04 m/s^2,
      # below the ~0.05-0.1 sustained-accel threshold. Runs BEFORE the slew (a +-0.3 step
      # traverses in ~50 ms -- effectively square at the wire).
      if (self.params.NIDEC_DITHER_AMP > 0.0 and CC.longActive and not gas_override_long
          and not self.descent and not lift_guard_active
          and not hud_control.leadVisible and abs(actuators.accel) < 0.15
          and CS.out.vEgo > 15.0 and not CS.out.brakePressed):
        if self.frame % max(int(self.params.NIDEC_DITHER_DWELL_S / DT_CTRL), 1) == 0:
          bit = ((self.dither_lfsr >> 6) ^ (self.dither_lfsr >> 5)) & 1   # x^7+x^6+1
          self.dither_lfsr = ((self.dither_lfsr << 1) | bit) & 0x7F
        pcm_off += self.params.NIDEC_DITHER_AMP * (1.0 if (self.dither_lfsr & 1) else -1.0)

      # asymmetric slew: fast UP so the PCM sees the full request promptly (it must also decide
      # on a downshift -- a slowly-growing request lets gear-hold hysteresis defer the kickdown),
      # slow DOWN to preserve the graded lift-off (pairs with DECEL_SOFT).
      pcm_off = rate_limit(pcm_off, self.last_pcm_off,
                           -self.params.NIDEC_MODEL_RATE * DT_CTRL, self.params.NIDEC_MODEL_RATE_UP * DT_CTRL)
      self.last_pcm_off = pcm_off
      self.po_filt += (DT_CTRL / 1.0) * (pcm_off - self.po_filt)
      pcm_speed = float(np.clip(CS.out.vEgo + pcm_off, 0.0, 100.0))
      pcm_accel = int(1.0 * self.params.NIDEC_GAS_MAX)
      # NIDEC_DESCENT_GAS_ZERO (default OFF, premise falsified -- see values.py): stock keeps
      # PCM_GAS at 198 during its own engine-brake downshifts, so zeroing diverges from stock.
      # Retained as a last-resort experiment behind the flag.
      if self.descent and self.params.NIDEC_DESCENT_GAS_ZERO:
        pcm_accel = 0
    elif (self.CP.carFingerprint in (CAR.ACURA_MDX_3G, CAR.ACURA_MDX_3G_MMR)):
      pcm_speed_V = [0.0,
                     np.clip(CS.out.vEgo - 2.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 2.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 20.0, 0.0, 100.0)]
      pcm_speed = float(np.interp(gas - brake, pcm_speed_BP, pcm_speed_V))
      pcm_accel = int(np.clip((accel / 1.44) / max_accel, 10.0 / self.params.NIDEC_GAS_MAX, 1.0) * self.params.NIDEC_GAS_MAX)
      if speed_control == 1 and CC.longActive:
        pcm_accel = 198
    else:
      pcm_speed_V = [0.0,
                     np.clip(CS.out.vEgo - 2.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 2.0, 0.0, 100.0),
                     np.clip(CS.out.vEgo + 5.0, 0.0, 100.0)]
      pcm_speed = float(np.interp(gas - brake, pcm_speed_BP, pcm_speed_V))
      pcm_accel = int(np.clip((accel / 1.44) / max_accel, 0.0, 1.0) * self.params.NIDEC_GAS_MAX)

    if not self.CP.openpilotLongitudinalControl:
      if self.frame % 2 == 0 and self.CP.carFingerprint not in HONDA_BOSCH_RADARLESS | HONDA_BOSCH_CANFD \
         and not self.CP.flags & HondaFlags.NIDEC:
        can_sends.append(hondacan.create_bosch_supplemental_1(self.packer, self.CAN))
      # If using stock ACC, spam cancel command to kill gas when OP disengages.
      if pcm_cancel_cmd:
        can_sends.append(hondacan.spam_buttons_command(self.packer, self.CAN, CruiseButtons.CANCEL, self.CP.carFingerprint))
      elif CC.cruiseControl.resume:
        can_sends.append(hondacan.spam_buttons_command(self.packer, self.CAN, CruiseButtons.RES_ACCEL, self.CP.carFingerprint))

    else:
      # Send gas and brake commands.
      if self.frame % 2 == 0:
        ts = self.frame * DT_CTRL

        if self.CP.carFingerprint in HONDA_BOSCH:
          if (accel < 0) and (CS.out.vEgo > 1e-3):
            brake_addon = self.brake_pid.update(error = accel - CS.out.aEgo, speed = CS.out.vEgo)
            targetaccel = min(accel,accel + brake_addon)
          else:
            self.brake_pid.reset()
            targetaccel = accel

          self.accel = float(np.clip(targetaccel, self.params.BOSCH_ACCEL_MIN, self.params.BOSCH_ACCEL_MAX))
          gas_pedal_force = self.accel + wind_brake_ms2 * self.windfactor + hill_brake

          # live-learn gas pedal adjustments when openpilot is controlling gas
          if (actuators.longControlState == LongCtrlState.pid) and (not CS.out.gasPressed):
            gas_error = self.accel - CS.out.aEgo
            if gas_error != 0.0 and gas_pedal_force > 0.0:
              if self.CP.carFingerprint == CAR.HONDA_INSIGHT: # Insight gas pedal reacts too slowly
                learn_speed = 150
              elif self.CP.carFingerprint in (CAR.ACURA_RDX_3G, CAR.ACURA_RDX_3G_MMR): # Prevent overreacting to turbo lag
                learn_speed = 300
              else:
                learn_speed = 50
              self.gasfactor = np.clip(self.gasfactor + gas_error / learn_speed * gas_pedal_force, 0.1, 3.0)
            if gas_error != 0.0 and (not CS.out.brakePressed) and (CS.out.vEgo > 0.0):
              if self.CP.carFingerprint in (CAR.ACURA_RDX_3G, CAR.ACURA_RDX_3G_MMR): # Faster reaction
                wind_learn_speed = 100
              else:
                wind_learn_speed = 1000
              wind_adjust = 1 + wind_brake_ms2 / wind_learn_speed
              self.windfactor = np.clip(self.windfactor * (wind_adjust if (gas_error > 0) else 1.0/wind_adjust), 0.1, 3.0)
            if gas_pedal_force <= 0.0: # don't reduce windfactor while braking, allow increases
              self.windfactor = max(self.windfactor, self.windfactor_before_brake)
            else:
              self.windfactor_before_brake = self.windfactor
            if gas_pedal_force >= self.params.BOSCH_ACCEL_MAX: # don't increase gasfactor nor windfactor at accel max, allow decreases
              self.gasfactor = min(self.gasfactor, self.gasfactor_before_gasmax)
              self.windfactor = min(self.windfactor, self.windfactor_before_gasmax)
            else:
              self.gasfactor_before_gasmax = self.gasfactor
              self.windfactor_before_gasmax = self.windfactor
          self.gas = float(np.interp(gas_pedal_force * self.gasfactor, self.params.BOSCH_GAS_LOOKUP_BP, self.params.BOSCH_GAS_LOOKUP_V))

          # limit gas ramp to 60 units per frame, matches stock.  Higher sometimes causes powertrain to ignore gas command.
          max_gas = max(60, self.bosch_last_gas + 60)
          self.gas = min(self.gas, max_gas)
          self.bosch_last_gas = self.gas

          stopping = actuators.longControlState == LongCtrlState.stopping
          self.stopping_counter = self.stopping_counter + 1 if stopping else 0
          can_sends.extend(hondacan.create_acc_commands(self.packer, self.CAN, CC.enabled, CC.longActive, self.accel, self.gas,
                                                        self.stopping_counter, self.CP.carFingerprint, gas_pedal_force))
        else:
          # Scale the aero-drag offset to the Odyssey's measured (smaller) coastdown so the
          # friction brake picks up where the gas-side ACC saturates (~-0.33 m/s2), instead of
          # the generic wind_brake holding it off until ~-0.46. Closes the moderate-decel dead-band.
          if self.CP.carFingerprint in HONDA_NIDEC_ALT_PCM_ACCEL:
            # aero + engine-brake already credited in the honest Plant-B demand above; the wind_brake
            # subtraction here was the 2.3x aero over-credit in the droop decomposition. Do not double-credit.
            apply_brake = np.clip(self.brake_last, 0.0, 1.0)
          else:
            apply_brake = np.clip(self.brake_last - wind_brake * self.params.NIDEC_BRAKE_WIND_FACTOR, 0.0, 1.0)
          apply_brake = int(np.clip(apply_brake * self.params.NIDEC_BRAKE_MAX, 0, self.params.NIDEC_BRAKE_MAX - 1))
          # Guardrail: never apply direct brake while ACC is requesting above-vEgo gas (pcm_off>0).
          # Normally unreachable after the 2.2g brake fix (pcm_off and brake are mutually exclusive),
          # but rate-limit lag and low-speed creep-brake can create simultaneous signals.
          if self.CP.carFingerprint in HONDA_NIDEC_ALT_PCM_ACCEL and self.last_pcm_off > 0:
            apply_brake = 0
          # POST-gate committed bite, v2 (2026-07-25 hardening). Two things at once, and the
          # ORDER MATTERS:
          #  * POST-gate -- a PRE-gate dwell does literally nothing (measured); the guardrail
          #    just chops the held value;
          #  * the hold's floor is the minimum FELT amplitude, a CONSTANT (`hold_floor`), never
          #    the window peak. v1 latched `max(hold_target, apply_brake)` and never let the
          #    floor fall, so the command floor for a whole 2 s dwell was the peak of the
          #    window: a transient spike became a full-amplitude 2 s bite (~4x the delivered
          #    speed loss), and at deep asks the peak-derived `hold_target` reached
          #    NIDEC_BRAKE_MAX = 256, one count above panda's HONDA_NIDEC_LONG_LIMITS.max_brake
          #    = 255, which makes `longitudinal_brake_checks` drop the WHOLE 0x1FA frame --
          #    brake command AND brake lights -- silently (nothing consumes safetyTxBlocked).
          #    Measured on the adversarial envelope: 579 of 950 sent frames dropped on an
          #    ordinary -3.0 m/s^2 stop approach.
          # The command therefore TRACKS the real demand upward and may relax back down to the
          # felt floor, but never below it and never off, for the dwell. Because the floor is a
          # constant 31 counts it cannot saturate the wire, cannot over-deliver by more than one
          # felt unit, and cannot re-kick the VSA pump when a dwell re-commits under a steady
          # demand (v1 walked its ramp back up through apply_brake every 2 s: 4 pump episodes
          # per 8 s hold against baseline's 1).
          # peak >= NIDEC_FRIC_MIN_AMP and duration >= NIDEC_BRAKE_MIN_DWELL still hold BY
          # CONSTRUCTION for any application whose amplitude comes from `friction_ms2` (the
          # commit is gated on friction_ms2 > 0, so the legacy sub-2.3 m/s `creep_brake` ramp --
          # which can cross the knee with NO friction demand -- can no longer commit a dwell;
          # its own 19-32 count command is untouched and is NOT covered by the felt floor).
          # At a dwell >= 1.5 s pulse trains are impossible (onsets can no longer be 0.2-1.5 s
          # apart). This block runs in the 50 Hz `frame % 2` branch, hence the 2 * DT_CTRL tick.
          if self.params.NIDEC_BRAKE_MIN_DWELL > 0.0 and \
             self.CP.carFingerprint in HONDA_NIDEC_ALT_PCM_ACCEL:
            # The ONE amplitude the hold can command: what the friction map returns at exactly
            # NIDEC_FRIC_MIN_AMP. ROUND UP -- int() truncation gave 31 counts = 0.196 m/s^2, i.e.
            # the floor sat one count BELOW the 0.20 threshold this whole change is built on, and
            # 0.196 is also the modal commanded amplitude, so by follow_score's decode roughly half
            # of all applications would have scored sub-feel (P6 sub-feel share 0.11 -> 0.29 vs
            # Adam's 0.03). ceil -> 32 counts = 0.206 m/s^2, which clears the threshold under every
            # decode in use (0.2063 plant model, 0.2071 follow_score.CB_TO_MS2). Both reverify
            # agents flagged this independently. At MIN_AMP = 0 it degenerates to the knee, i.e.
            # "keep the pads engaged for the dwell, with no felt floor".
            hold_floor = math.ceil(np.clip(self.params.NIDEC_FRIC_MIN_AMP / self.params.NIDEC_MODEL_BRAKE_PLANT
                                           + self.params.NIDEC_MODEL_BRAKE_KNEE, 0.0, 1.0)
                                   * self.params.NIDEC_BRAKE_MAX)
            # the hydraulic preload: the first count that can produce any decel at all
            knee_counts = round(self.params.NIDEC_MODEL_BRAKE_KNEE * self.params.NIDEC_BRAKE_MAX)
            # "the plan wants to go": a THRESHOLD plus a debounce, not a bare `accel > 0`, which
            # would chop the bite on every zero crossing. The published rejection of an early
            # release was measured on traces whose dwell overhang never exceeded ask +0.086 --
            # below this threshold, so it does not fire there at all -- while the reviewers
            # measured the hold surviving a ramp to +0.6 with pcm_off railed at 5.0 m/s.
            if self.params.NIDEC_BRAKE_HOLD_RELEASE_A > 0.0 and \
               actuators.accel >= self.params.NIDEC_BRAKE_HOLD_RELEASE_A:
              self.brake_hold_pos_frames += 1
            else:
              self.brake_hold_pos_frames = 0
            wants_to_go = self.brake_hold_pos_frames >= \
                          max(1, int(self.params.NIDEC_BRAKE_HOLD_RELEASE_T / (2 * DT_CTRL)))
            # Releases: disengagement / gas-override handoff (longActive is False there too, and
            # panda blocks 0x1FA while the pedal is down); the descent latch, whose entire point
            # is that the PCM downshift serves the band and friction stays OFF (friction_ms2 is
            # forced to 0 in-latch, so holding a bite through it inverts the feature); and the
            # plan asking to accelerate.
            if (not CC.longActive) or self.descent or wants_to_go:
              self.brake_hold_frames, self.brake_hold_ramp = 0, 0.0
            else:
              if apply_brake > knee_counts and friction_ms2 > 0.0 and self.brake_hold_frames <= 0:
                self.brake_hold_frames = int(self.params.NIDEC_BRAKE_MIN_DWELL / (2 * DT_CTRL))
                # start the floor where the command already is, and do NOT advance it on this
                # frame, so the commit itself is a strict no-op on the wire: the floor can only
                # ever stop the command falling, never step it up
                self.brake_hold_ramp = float(min(apply_brake, hold_floor))
              else:
                self.brake_hold_frames = max(self.brake_hold_frames - 1, 0)
                if self.brake_hold_frames > 0:
                  # same up-rate as the demand path's own rate limit (line ~418), so the hold
                  # adds no jerk the friction channel could not already produce
                  self.brake_hold_ramp = min(float(hold_floor), self.brake_hold_ramp
                                             + 3.0 * (2 * DT_CTRL) * self.params.NIDEC_BRAKE_MAX)
              if self.brake_hold_frames > 0:
                apply_brake = max(apply_brake, int(self.brake_hold_ramp))
              else:
                self.brake_hold_ramp = 0.0
          # THE wire clamp. Last thing to touch apply_brake before the pump hysteresis, the
          # packer, self.apply_brake_last and self.brake, so no present or future path can put a
          # value on 0x1FA that panda will reject. `NIDEC_BRAKE_MAX - 1` == panda's max_brake
          # (255); the check there is `desired_brake > max_brake` -> tx = false -> the frame is
          # dropped whole. A no-op at every knob's off value (the clip at line ~749 already
          # bounds the pre-existing path), which is what keeps the exact-revert golden valid.
          apply_brake = int(np.clip(apply_brake, 0, self.params.NIDEC_BRAKE_MAX - 1))
          pump_on, self.last_pump_ts = brake_pump_hysteresis(apply_brake, self.apply_brake_last, self.last_pump_ts, ts)

          pcm_override = True
          can_sends.append(hondacan.create_brake_command(self.packer, self.CAN, apply_brake, pump_on,
                                                         pcm_override, pcm_cancel_cmd, alert_fcw,
                                                         self.CP.carFingerprint, CS.stock_brake, self.CP_SP))
          self.apply_brake_last = apply_brake
          self.brake = apply_brake / self.params.NIDEC_BRAKE_MAX

          gas_error = actuators.accel - CS.out.aEgo
          if (not CS.out.gasPressed) and (actuators.longControlState == LongCtrlState.pid) and self.CP_SP.enableGasInterceptor:
            if gas_error != 0.0 and gas > 0.0:
              self.gasfactor = np.clip(self.gasfactor + gas_error / 150 * (gas * 4.8), 0.1, 3.0)
            if gas_error != 0.0 and (not CS.out.brakePressed) and (CS.out.vEgo > 0.0):
              wind_adjust = 1 + (wind_brake * 4.8) / 1000
              self.windfactor = np.clip(self.windfactor * (wind_adjust if (gas_error > 0) else 1.0/wind_adjust), 0.1, 5.0)
            if gas <= 0.0: # don't reduce windfactor while braking, allow increases
              self.windfactor = max(self.windfactor, self.windfactor_before_brake)
            else:
              self.windfactor_before_brake = self.windfactor

          can_sends.extend(GasInterceptorCarController.update(self, CC, CS, gas * self.gasfactor, brake, wind_brake, self.packer, self.frame))

    # Send dashboard UI commands.
    if self.frame % 10 == 0:
      if CC.longActive and (self.CP.carFingerprint in (CAR.ACURA_MDX_3G, CAR.ACURA_MDX_3G_MMR)):
        # standstill disengage
        if (accel >= 0.01) and (CS.out.vEgo < 4.0) and (pcm_speed < 25.0 / 3.6):
          pcm_speed = 25.0 / 3.6

      if self.CP.openpilotLongitudinalControl:
        # On Nidec, this also controls longitudinal positive acceleration
        can_sends.append(hondacan.create_acc_hud(self.packer, self.CAN.pt, self.CP, CC.enabled, pcm_speed, pcm_accel,
                                                 hud_control, hud_v_cruise, CS.is_metric, CS.acc_hud, speed_control))

      steering_available = CS.out.cruiseState.available and CS.out.vEgo > max(self.params.STEER_GLOBAL_MIN_SPEED, self.CP.minSteerSpeed)
      reduced_steering = CS.out.steeringPressed
      steer_maxed = abs(apply_torque) >= self.params.STEER_MAX
      can_sends.extend(hondacan.create_lkas_hud(self.packer, self.CAN.lkas, self.CP, hud_control, CC.latActive,
                                                steering_available, reduced_steering, alert_steer_required, CS.lkas_hud, self.dashed_lanes,
                                                steer_maxed))

      if self.CP.openpilotLongitudinalControl:
        # TODO: combining with create_acc_hud block above will change message order and will need replay logs regenerated
        if self.CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS):
          can_sends.append(hondacan.create_radar_hud(self.packer, self.CAN.pt))
        if self.CP.carFingerprint == CAR.HONDA_CIVIC_BOSCH:
          can_sends.append(hondacan.create_legacy_brake_command(self.packer, self.CAN.pt))
        if self.CP.carFingerprint not in HONDA_BOSCH:
          self.speed = pcm_speed
          if not self.CP_SP.enableGasInterceptor:
            self.gas = pcm_accel / self.params.NIDEC_GAS_MAX

    # Intelligent Cruise Button Management
    can_sends.extend(IntelligentCruiseButtonManagementInterface.update(self, CC_SP, self.packer, self.frame,
                                                                       self.last_button_frame, self.CAN))

    new_actuators = actuators.as_builder()
    new_actuators.speed = self.speed
    new_actuators.accel = self.accel
    new_actuators.gas = float(self.gasfactor)
    new_actuators.brake = float(self.windfactor)
    new_actuators.torque = self.last_torque
    new_actuators.torqueOutputCan = apply_torque

    if self.frame % 6000 == 0:
      self.param_writer.put_many({
        "HondaGasFactorParams": self.gasfactor,
        "HondaWindFactorParams": self.windfactor,
      })

    self.frame += 1
    return new_actuators, can_sends
