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
    self.a_des_lp = 0.0      # servo-aware FF: low-pass of a_des for the build-vs-ease latch
    self.po_build = False    # servo-aware FF: True while the accel request is building (onset)
    self.dbo_sus = 0         # DBO: frames the demand has exceeded the nudge-cap reach (sustain gate)
    self.knee_guard = True   # DBO knee guard: True while demand is sub-threshold (guarded by default)
    self.gas_override_linger = 0  # frames to hold the override path through the release handback

    # Tier-1 kickdown-surge trim state (see NIDEC_TRIM_* in values.py)
    self.trim_frames = 0          # frames remaining in the trim/recovery window
    self.last_tgt_gear = 0        # last valid TRANS_TARGET_GEAR (0 = unknown)
    self.po_filt = 0.0            # ~1s low-pass of pcm_off (the TCU schedule sees history)

    # Descent-mode latch state (NIDEC_DESCENT_* in values.py, SPEC_descent_mode_2026-07-11)
    self.descent = False         # engine-brake-first latch
    self.descent_bte = False     # band-top exited: re-entry only below BAND_REARM (sawtooth bound)
    self.descent_pitch_lp = 0.0  # ~1s LP of pitch for the latch only (FF paths keep raw pitch)

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
    # ~1s LP of pitch for the descent-mode latch only (runs every frame so the filter is
    # already settled when the latch conditions are first evaluated)
    self.descent_pitch_lp += (DT_CTRL / self.params.NIDEC_DESCENT_PITCH_TAU) * (self.pitch - self.descent_pitch_lp)

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
        # Descent-mode latch (SPEC_descent_mode_2026-07-11): engine-brake-first descents. All
        # gates hysteretic. v_cruise = HUD set speed (m/s; guarded by speedVisible -- a stale
        # 255 also fails the band by construction). The a_des_b gate is the wire to the
        # planner's DESCENT_* tolerance floor: while the planner tolerates the band, a_des_b
        # rides ~-0.1..-0.3 even at -5.6% grade; any released lead/curve/e2e demand drives it
        # past ADES_MIN -> same-frame release and the friction cover below serves unchanged.
        if self.params.NIDEC_DESCENT and not gas_override_long:
          v_err = CS.out.vEgo - hud_control.setSpeed  # >0 = over set (m/s)
          valid = hud_control.speedVisible and CS.out.vEgo > self.params.NIDEC_DESCENT_V_MIN
          pitch_ok = self.descent_pitch_lp < (self.params.NIDEC_DESCENT_PITCH_OFF if self.descent
                                              else self.params.NIDEC_DESCENT_PITCH_ON)
          ades_ok = a_des_b > (self.params.NIDEC_DESCENT_ADES_MIN if self.descent
                               else self.params.NIDEC_DESCENT_ADES_REARM)
          if self.descent:
            band_ok = self.params.NIDEC_DESCENT_BAND_LOW < v_err < self.params.NIDEC_DESCENT_BAND_TOP
            if not band_ok and v_err >= self.params.NIDEC_DESCENT_BAND_TOP:
              self.descent_bte = True  # band-top exit: friction trims; re-arm only below REARM
            self.descent = valid and pitch_ok and band_ok and ades_ok
          else:
            # First entry is allowed anywhere in the band: today's steep-descent friction
            # equilibrium sits at ~+2 kph (knee under-delivery), between REARM and TOP -- an
            # entry ceiling at REARM could never engage there (measured: 370/975 grade-friction
            # frames, descent_check.py). REARM only bounds the post-band-top trim sawtooth.
            if self.descent_bte and v_err < self.params.NIDEC_DESCENT_BAND_REARM:
              self.descent_bte = False
            band_hi = self.params.NIDEC_DESCENT_BAND_REARM if self.descent_bte else self.params.NIDEC_DESCENT_BAND_TOP
            self.descent = (valid and pitch_ok and ades_ok and
                            self.params.NIDEC_DESCENT_BAND_LOW < v_err < band_hi)
        else:
          self.descent = False
          self.descent_bte = False
        # In-latch the band IS the demand server (PCM engine brake via its own downshift);
        # past-band / released frames are byte-identical to baseline.
        friction_ms2 = 0.0 if self.descent else max(0.0, -a_des_b - eb_credit - wind_ms2)
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
          if a_des < uncap_a - self.params.NIDEC_DBO_UNCAP_A_HYST:
            self.knee_guard = True
          elif a_des > uncap_a:
            self.knee_guard = False
          if self.knee_guard:
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

      # Descent-mode anchor (SPEC_descent_mode_2026-07-11): while latched, present the PCM the
      # stock descent signal -- PCM_SPEED held at set-BIAS while vEgo grows = growing overspeed
      # error (stock's own downshift trigger, pre-loaded ~2 kph so it fires earlier than
      # stock's +1..+6 kph tolerance). clip to [PCM_OFF_MIN, 0]: never outside the envelope we
      # already command daily, never adds gas (positive asks stay on the governed path above).
      # The existing slew below rate-limits the entry step; on release the normal FF value
      # re-enters through the same slew -- no discontinuity either direction.
      if self.descent:
        pcm_off = float(np.clip((hud_control.setSpeed - self.params.NIDEC_DESCENT_BIAS) - CS.out.vEgo,
                                self.params.NIDEC_MODEL_PCM_OFF_MIN, 0.0))

      # asymmetric slew: fast UP so the PCM sees the full request promptly (it must also decide
      # on a downshift -- a slowly-growing request lets gear-hold hysteresis defer the kickdown),
      # slow DOWN to preserve the graded lift-off (pairs with DECEL_SOFT).
      pcm_off = rate_limit(pcm_off, self.last_pcm_off,
                           -self.params.NIDEC_MODEL_RATE * DT_CTRL, self.params.NIDEC_MODEL_RATE_UP * DT_CTRL)
      self.last_pcm_off = pcm_off
      self.po_filt += (DT_CTRL / 1.0) * (pcm_off - self.po_filt)
      pcm_speed = float(np.clip(CS.out.vEgo + pcm_off, 0.0, 100.0))
      pcm_accel = int(1.0 * self.params.NIDEC_GAS_MAX)
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
