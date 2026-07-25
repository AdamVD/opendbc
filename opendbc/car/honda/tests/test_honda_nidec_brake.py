"""NIDEC_ALT (Odyssey) friction-channel invariants.

These lock down the 2026-07-25 brake-channel ship candidate: `NIDEC_GRADE_CREDIT_CLAMP`
+ `NIDEC_FRIC_MIN_AMP` + `NIDEC_BRAKE_MIN_DWELL` (+ the committed bite's release knobs).
The behavioural requirement is Adam's, in his words: "no spammy light braking EVER".
Made concrete, for any application whose amplitude comes from `friction_ms2`:

  * it may never peak below the count floor implied by `NIDEC_FRIC_MIN_AMP`, and
  * it may never be shorter than `NIDEC_BRAKE_MIN_DWELL` unless legally released,

both BY CONSTRUCTION rather than by tuning.  The shipping code before this change
produced 621 applications violating one of those two over 595.6 engaged minutes.

⚠ EVERY INVARIANT HERE IS TWO-SIDED, and that is the whole lesson of this file.  The
first cut had floors only.  It shipped a hold that saturated `apply_brake` at 256 -- one
count above panda's `HONDA_NIDEC_LONG_LIMITS.max_brake` -- which makes panda drop the
ENTIRE 0x1FA frame (brake command AND brake lights), silently, from an ask of about
-2.4 m/s^2 downhill.  It survived every sweep because every sweep, the golden sequence
and the whole 100 Hz corpus stopped at an ask of -1.45 while the car authorises -4.0.
So: the ceiling tests are asserted on the PACKED CAN frame decoded with panda's own bit
math, panda's tx hook is modelled and a DROPPED FRAME is the failure, and every sweep
spans the full authorised band.

Every knob is also an exact-revert gate: with all of them at their off values the
controller must be bit-identical to the prior behaviour.  That is `test_exact_revert`,
and it is the most important test in this file -- the candidate was scored in a harness
whose enable gates read a policy object, so the literal `self.params.NIDEC_*` text that
ships here is executed for the first time by these tests.  It compares against a golden
trace captured from the pre-change controller (see `nidec_brake_revert_golden.npz` and
`gen_nidec_brake_golden.py`).
"""
import math
import os

import numpy as np
import pytest

from opendbc.car import Bus, DT_CTRL, structs
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda import carcontroller as carcontroller_mod
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.values import CAR, CarControllerParams

P = CarControllerParams

# apply_brake counts <-> m/s^2, using the identified brake plant.  `apply_brake` is the
# 0..NIDEC_BRAKE_MAX COMPUTER_BRAKE value that actually reaches 0x1FA.
KNEE_COUNTS = round(P.NIDEC_MODEL_BRAKE_KNEE * P.NIDEC_BRAKE_MAX)   # 13, the hydraulic preload
COUNT_MS2 = P.NIDEC_MODEL_BRAKE_PLANT / P.NIDEC_BRAKE_MAX           # m/s^2 per count

# The structural amplitude floor: the smallest apply_brake the committed-bite hold can
# latch, i.e. what the hold floor evaluates to at exactly NIDEC_FRIC_MIN_AMP.  Rounded UP,
# so the realised floor clears the nominal 0.20 m/s^2 instead of landing a count under it
# (int() truncation gave 31 = 0.196, which made the floor itself score sub-feel).
MIN_COUNTS = math.ceil(np.clip(P.NIDEC_FRIC_MIN_AMP / P.NIDEC_MODEL_BRAKE_PLANT
                               + P.NIDEC_MODEL_BRAKE_KNEE, 0.0, 1.0) * P.NIDEC_BRAKE_MAX)

GOLDEN = os.path.join(os.path.dirname(__file__), 'nidec_brake_revert_golden.npz')

# ---------------------------------------------------------------------------------------
# panda's own arithmetic, reimplemented from opendbc/safety/modes/honda.h so the ceiling
# assertions below are made against the WIRE, not against an intermediate Python variable.
# The 2026-07-25 blocker got in precisely because a new code path bypassed the variable
# everyone was watching: `hold_target` clipped to NIDEC_BRAKE_MAX (256) where every other
# path clips to NIDEC_BRAKE_MAX - 1, and panda's HONDA_NIDEC_LONG_LIMITS.max_brake is 255.
# `longitudinal_brake_checks` then sets tx = false, which drops the WHOLE 0x1FA frame --
# brake command and brake lights -- and nothing in openpilot consumes safetyTxBlocked, so
# the failure is silent.
# ---------------------------------------------------------------------------------------
PANDA_MAX_BRAKE = 255           # HONDA_NIDEC_LONG_LIMITS.max_brake
PANDA_MAX_GAS = 198             # HONDA_NIDEC_LONG_LIMITS.max_gas (0xc6)
PANDA_INACTIVE_SPEED = 0        # HONDA_NIDEC_LONG_LIMITS.inactive_speed
ADDR_BRAKE = 0x1FA
ADDR_ACC_HUD = 0x30C


def panda_decode_brake(data) -> int:
  """`honda_brake = (GET_BYTE(0) << 2) + ((GET_BYTE(1) >> 6) & 0x3U)` (honda.h)."""
  return (data[0] << 2) + ((data[1] >> 6) & 0x3)


def panda_decode_acc_hud(data):
  """0x30C: pcm_speed is bytes 0-1, pcm_gas is byte 2."""
  return ((data[0] << 8) | data[1]), data[2]


def panda_brake_tx(data, controls_allowed=True, gas_pressed_prev=False):
  """`longitudinal_brake_checks`: reject on > max_brake, and on any non-zero brake when
  `get_longitudinal_allowed()` (controls_allowed and not gas_pressed_prev) is false."""
  hb = panda_decode_brake(data)
  if hb > PANDA_MAX_BRAKE:
    return False, hb, f'brake {hb} > max_brake {PANDA_MAX_BRAKE}'
  if (not (controls_allowed and not gas_pressed_prev)) and hb != 0:
    return False, hb, '!longitudinal_allowed and brake != 0'
  return True, hb, ''


def panda_acc_hud_tx(data, controls_allowed=True):
  """0x30C: `safety_max_limit_check(pcm_gas, max_gas, 0)` is `val > MAX || val < MIN`, so
  198 passes with ZERO counts of margin and 199 would drop the frame."""
  pcm_speed, pcm_gas = panda_decode_acc_hud(data)
  if (not controls_allowed) and pcm_speed != PANDA_INACTIVE_SPEED:
    return False, pcm_gas, '!allowed and pcm_speed != inactive'
  if not (pcm_gas == 0 or (controls_allowed and 0 <= pcm_gas <= PANDA_MAX_GAS)):
    return False, pcm_gas, f'pcm_gas {pcm_gas} > max_gas {PANDA_MAX_GAS}'
  return True, pcm_gas, ''

# the aero/coastdown table and engine-brake credit the NIDEC_ALT friction demand uses
_WIND_BP = [0.0, 13.4, 22.4, 31.3, 40.2]
_WIND_V = [0.000, 0.049, 0.136, 0.267, 0.441]


def counts_to_ms2(counts: int) -> float:
  return max(0.0, counts - KNEE_COUNTS) * COUNT_MS2


def flat_friction_demand(accel: float, v_ego: float) -> float:
  """The controller's `fric_demand`, for pitch == 0 with the lift guard and descent off."""
  return max(0.0, -accel - P.NIDEC_MODEL_ENGINE_BRAKE - float(np.interp(v_ego, _WIND_BP, _WIND_V)))


def demand_counts(fric_ms2: float) -> int:
  """`hold_target`: the apply_brake the demand asks for, before hysteresis/rate limiting."""
  if fric_ms2 <= 0.0:
    return 0
  return int(np.clip(fric_ms2 / P.NIDEC_MODEL_BRAKE_PLANT + P.NIDEC_MODEL_BRAKE_KNEE,
                     0.0, 1.0) * P.NIDEC_BRAKE_MAX)


class _Zero(dict):
  def __missing__(self, _k):
    return 0


class _CarStateStub:
  """The subset of the CarState wrapper the Honda CarController touches."""

  def __init__(self, v_ego, engine_rpm=1500.0, xmission_speed=100.0, gas_pressed=False):
    self.out = structs.CarState()
    self.out.vEgo = v_ego
    self.out.aEgo = 0.0
    self.out.gasPressed = gas_pressed
    self.out.brakePressed = False
    self.out.cruiseState.available = True
    self.v_cruise_factor = 1.0
    # gear ratio = engine_rpm / xmission_speed drives the low-gear lift guard latch:
    # ON above NIDEC_LIFT_RATIO_ON (24), OFF below NIDEC_LIFT_RATIO_OFF (22.5).
    # The default 1500/100 = 15 leaves the guard off (normal cruise).
    self.engine_rpm = engine_rpm
    self.xmission_speed = xmission_speed
    self.is_metric = False
    self.acc_hud = _Zero()
    self.lkas_hud = _Zero()
    self.stock_brake = 0


class _NoParamWriter:
  """`HondaParamWriter` starts a daemon thread and holds a `Params()` handle per
  CarController.  These sweeps build hundreds of controllers in one process, which
  exhausts threads/file handles and wedges the whole run; the writer only persists the
  learned gas/wind factors every 6000 frames and has no effect on any command."""

  def __init__(self, *a, **k):
    pass

  def put_many(self, values):
    pass


def build_carcontroller(cls, fingerprint, CP, CP_SP):
  """Construct `cls` with the param writer stubbed out for the duration."""
  orig = carcontroller_mod.HondaParamWriter
  carcontroller_mod.HondaParamWriter = _NoParamWriter
  try:
    return cls({Bus.pt: fingerprint.config.dbc_dict[Bus.pt]}, CP, CP_SP)
  finally:
    carcontroller_mod.HondaParamWriter = orig


class Rig:
  """A real Honda CarController driven frame by frame with a synthetic CarState.

  `step()` returns the wire brake command (the value packed into 0x1FA, latched in the
  50 Hz block), the wire pcm_off, and `brake_last` -- the pre-guardrail brake fraction,
  which is the friction demand path's own output and updates every frame.
  """

  def __init__(self, fingerprint=CAR.HONDA_ODYSSEY, **params):
    CP = interfaces[fingerprint].get_non_essential_params(fingerprint)
    CP.openpilotLongitudinalControl = True
    CP_SP = structs.CarParamsSP()
    self.cc = build_carcontroller(CarController, fingerprint, CP, CP_SP)
    # per-instance parameter overrides (the class attributes hold the shipping values)
    for k, v in params.items():
      assert hasattr(self.cc.params, k), k
      setattr(self.cc.params, k, v)
    self.now_nanos = 0
    # WIRE record: the PACKED 0x1FA / 0x30C payloads decoded with panda's own bit math, plus
    # panda's verdict.  `wire` is appended to on every 100 Hz frame; 0x1FA is packed at 50 Hz
    # and 0x30C at 10 Hz, so the last packed value is held between updates (which is what the
    # car sees).  Fields: (brake_counts, brake_tx_ok, pcm_gas, hud_tx_ok, brake_frame_sent).
    self.wire = []
    self._w = [0, 1, 0, 1, 0]

  def step(self, accel, v_ego=30.0, pitch=0.0, long_active=True, enabled=True,
           lead_visible=False, set_speed=32.0, engine_rpm=1500.0, xmission_speed=100.0,
           gas_pressed=False, n=1):
    out = []
    for _ in range(n):
      CS = _CarStateStub(v_ego, engine_rpm, xmission_speed, gas_pressed)
      CC = structs.CarControl()
      CC.enabled = enabled
      CC.longActive = long_active
      CC.orientationNED = [0.0, pitch, 0.0]
      CC.actuators.accel = accel
      CC.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
      CC.hudControl.speedVisible = True
      CC.hudControl.setSpeed = set_speed
      CC.hudControl.leadVisible = lead_visible
      _act, can_sends = self.cc.update(CC.as_reader(), structs.CarControlSP(), CS, self.now_nanos)
      self.now_nanos += int(DT_CTRL * 1e9)
      self._w[4] = 0
      for m in can_sends:
        addr, data = int(m[0]), bytes(m[-2] if len(m) == 4 else m[1])
        if addr == ADDR_BRAKE:
          ok, hb, _why = panda_brake_tx(data, controls_allowed=enabled, gas_pressed_prev=gas_pressed)
          self._w[0], self._w[1], self._w[4] = hb, int(ok), 1
        elif addr == ADDR_ACC_HUD:
          ok, pg, _why = panda_acc_hud_tx(data, controls_allowed=enabled)
          self._w[2], self._w[3] = pg, int(ok)
      self.wire.append(tuple(self._w))
      out.append((int(self.cc.apply_brake_last), float(self.cc.last_pcm_off), float(self.cc.brake_last)))
    return out if n > 1 else out[0]

  def wire_trace(self):
    return np.array(self.wire, dtype=np.int64)


def revert_sequence(n=3000):
  """The deterministic input sequence behind the golden trace.

  ⚠ 2026-07-25 hardening: this used to span only `accel` in [-1.2, +0.6] and `v_ego` in
  [15, 35].  That is why the exact-revert golden, every parameter sweep and the whole
  100 Hz corpus missed a saturation that starts at an ask of -2.4 .. -3.0.  It now spans
  the FULL authorised command band -- `actuators.accel` is clipped only by
  `get_pid_accel_limits` -> (NIDEC_ACCEL_MIN, ...) = -4.0, and the PID genuinely occupies
  [-4.0, -3.5) because the planner's own a_min is the shallower ACCEL_MIN = -3.5 -- plus
  standstill/creep speeds, the low-gear lift guard, gas-override windows and a lead.
  Imported by the golden generator, so the generator and the test can never drift.
  """
  rng = np.random.default_rng(0xB4A4E)
  i = 0
  while i < n:
    # PIECEWISE CONSTANT, not i.i.d. per frame: `brake_last` is rate-limited to 7.68 counts
    # per 100 Hz frame, so a per-frame random ask can never drive the command deep and the
    # golden would cover only the shallow band again -- the exact mistake this is fixing.
    seg = int(rng.integers(5, 140))
    kw = {'accel': float(rng.uniform(P.NIDEC_ACCEL_MIN, 2.0)),
          'pitch': float(rng.uniform(-0.10, 0.10)),
          'v_ego': float(rng.uniform(0.0, 40.0)),
          'set_speed': float(rng.uniform(20.0, 36.0)),
          'lead_visible': bool(rng.integers(0, 2)),
          # ratio = rpm/kph; > NIDEC_LIFT_RATIO_ON (24) latches the low-gear lift guard
          'engine_rpm': float(rng.choice([1500.0, 2800.0])),
          'gas_pressed': bool(rng.random() < 0.08),
          'long_active': bool(rng.random() < 0.88)}
    for _ in range(min(seg, n - i)):
      yield dict(kw)
      i += 1


def run_sequence(rig, n=3000):
  return np.array([rig.step(**kw) for kw in revert_sequence(n)], dtype=np.float64)


def brake_events(trace):
  """[(start_idx, end_idx_exclusive, peak_counts)] for each contiguous run of brake."""
  ev, start = [], None
  for i, ab in enumerate(trace):
    if ab > KNEE_COUNTS and start is None:
      start = i
    elif ab <= KNEE_COUNTS and start is not None:
      ev.append((start, i, max(trace[start:i])))
      start = None
  if start is not None:
    ev.append((start, len(trace), max(trace[start:])))
  return ev


# --------------------------------------------------------------------------------------
# exact-revert gate
# --------------------------------------------------------------------------------------

class TestExactRevert:
  def test_exact_revert(self):
    """All three knobs at their off values == the pre-change controller, frame for frame.

    The reference is a golden trace captured from the controller as it stood before this
    change (`gen_nidec_brake_golden.py`, which execs the pre-change source out of git).
    """
    ref = np.load(GOLDEN)['trace']
    got = run_sequence(Rig(NIDEC_GRADE_CREDIT_CLAMP=False, NIDEC_FRIC_MIN_AMP=0.0,
                           NIDEC_BRAKE_MIN_DWELL=0.0), len(ref))
    assert got.shape == ref.shape
    bad = np.flatnonzero(np.any(got != ref, axis=1))
    msg = f"{len(bad)}/{len(ref)} frames differ from the pre-change controller"
    assert len(bad) == 0, f"{msg}; first at {bad[:5]}: {got[bad[:3]]} vs {ref[bad[:3]]}"

  def test_golden_is_not_vacuous(self):
    """The shipping configuration must actually differ from the golden, or the exact-revert
    test proves nothing."""
    ref = np.load(GOLDEN)['trace']
    got = run_sequence(Rig(), len(ref))
    assert np.any(got != ref), "the ship configuration is indistinguishable from the revert"

  def test_golden_covers_the_deep_regime(self):
    """The reason this whole class of bug survived: the OLD golden sequence spanned an ask
    of [-1.2, +0.6] and never got past ~76 counts, so an exact-revert test over it could
    not see a saturation that starts at -2.4.  The sequence must reach the pre-existing
    path's structural maximum (253 counts -- the -0.01 brake hysteresis caps it there)."""
    ref = np.load(GOLDEN)['trace']
    assert ref[:, 0].max() >= 250, f"golden only reaches {ref[:, 0].max():.0f} counts"
    asks = np.array([kw['accel'] for kw in revert_sequence(len(ref))])
    assert asks.min() <= P.NIDEC_ACCEL_MIN + 0.05, asks.min()
    assert asks.max() >= 1.5, asks.max()

  @pytest.mark.parametrize("knob,off", [("NIDEC_GRADE_CREDIT_CLAMP", False),
                                        ("NIDEC_FRIC_MIN_AMP", 0.0),
                                        ("NIDEC_BRAKE_MIN_DWELL", 0.0),
                                        ("NIDEC_BRAKE_HOLD_RELEASE_A", 0.0)])
  def test_each_knob_is_load_bearing(self, knob, off):
    """Each knob on its own must move the wire, so none of them is dead code.

    (MIN_AMP is exercised with the dwell also on -- the staging lock forbids the other
    combination, and it is the arm measured worse than shipping on both drives.  The
    release knobs are exercised on top of the dwell, which is the only thing they gate.)
    """
    base = {"NIDEC_GRADE_CREDIT_CLAMP": False, "NIDEC_FRIC_MIN_AMP": 0.0,
            "NIDEC_BRAKE_MIN_DWELL": 0.0, "NIDEC_BRAKE_HOLD_RELEASE_A": 0.0}
    on = dict(base)
    on[knob] = getattr(P, knob)
    if knob in ("NIDEC_FRIC_MIN_AMP", "NIDEC_BRAKE_HOLD_RELEASE_A"):
      on["NIDEC_BRAKE_MIN_DWELL"] = P.NIDEC_BRAKE_MIN_DWELL
    if knob == "NIDEC_BRAKE_HOLD_RELEASE_A":
      base["NIDEC_BRAKE_MIN_DWELL"] = P.NIDEC_BRAKE_MIN_DWELL
    a = run_sequence(Rig(**base))
    b = run_sequence(Rig(**on))
    assert np.any(a != b), f"{knob} = {getattr(P, knob)} changed nothing"

  def test_release_debounce_is_load_bearing(self):
    """RELEASE_T is not covered by the sweep above (it only shifts WHEN the release fires),
    so pin it on the input it exists for: a chattering ask whose positive half-cycle is
    shorter than the debounce must not be able to break the committed bite."""
    def chatter(**knobs):
      rig = Rig(**knobs)
      rig.step(0.0, v_ego=30.0, n=100)
      tr = []
      for _ in range(12):
        tr += [ab for ab, _po, _bl in rig.step(-0.80, v_ego=30.0, n=20)]
        tr += [ab for ab, _po, _bl in rig.step(+0.30, v_ego=30.0, n=20)]
      tr += [ab for ab, _po, _bl in rig.step(+0.30, v_ego=30.0, n=300)]
      return brake_events(tr)
    short = chatter(NIDEC_BRAKE_HOLD_RELEASE_T=0.20)     # == the chatter half-cycle
    ship = chatter()                                     # 0.35 s
    assert min((e - s) * DT_CTRL for s, e, _p in short) < 0.5, short
    assert len(ship) == 1 and (ship[0][1] - ship[0][0]) * DT_CTRL >= \
           P.NIDEC_BRAKE_MIN_DWELL - 2 * DT_CTRL, ship


# --------------------------------------------------------------------------------------
# the hard invariant: amplitude and dwell
# --------------------------------------------------------------------------------------

class TestFrictionInvariant:
  def test_sub_feel_demand_never_reaches_the_wire(self):
    rig = Rig()
    assert flat_friction_demand(-0.10, 30.0) < P.NIDEC_FRIC_MIN_AMP   # premise
    trace = [ab for ab, _po, _bl in rig.step(-0.10, v_ego=30.0, n=300)]
    assert max(trace) == 0, f"sub-feel ask produced {max(trace)} counts of brake"

  def test_min_amplitude_swept(self):
    """No commanded application may peak below the MIN_AMP count floor, at any ask depth."""
    assert MIN_COUNTS == 32                          # cross-check vs the scored bench
    # the point of the floor: it must actually clear the felt threshold it is named after
    assert counts_to_ms2(MIN_COUNTS) >= P.NIDEC_FRIC_MIN_AMP
    assert counts_to_ms2(MIN_COUNTS) == pytest.approx(0.2063, abs=1e-3)
    bad = []
    for ask in np.arange(-1.20, -0.005, 0.01):
      for v in (18.0, 26.0, 34.0):
        rig = Rig()
        trace = [ab for ab, _po, _bl in rig.step(float(ask), v_ego=v, n=250)]
        for _s, _e, peak in brake_events(trace):
          if peak < MIN_COUNTS:
            bad.append((round(float(ask), 3), v, peak, round(counts_to_ms2(peak), 4)))
    assert not bad, f"applications below the structural floor: {bad[:10]}"

  def test_min_amplitude_swept_baseline_violates(self):
    """The same sweep on the pre-change configuration DOES produce sub-floor bites --
    otherwise the sweep above is not testing anything."""
    bad = 0
    for ask in np.arange(-1.20, -0.005, 0.01):
      rig = Rig(NIDEC_GRADE_CREDIT_CLAMP=False, NIDEC_FRIC_MIN_AMP=0.0, NIDEC_BRAKE_MIN_DWELL=0.0)
      trace = [ab for ab, _po, _bl in rig.step(float(ask), v_ego=26.0, n=250)]
      bad += sum(1 for _s, _e, peak in brake_events(trace) if peak < MIN_COUNTS)
    assert bad > 0

  def test_min_dwell_short_transient(self):
    """A 0.1 s demand -- the median bite duration of the shipping code on the bad leg --
    must be promoted to a full committed bite instead of a flicker."""
    rig = Rig()
    trace = [ab for ab, _po, _bl in rig.step(0.0, v_ego=30.0, n=50)]
    trace += [ab for ab, _po, _bl in rig.step(-0.80, v_ego=30.0, n=10)]   # the transient
    trace += [ab for ab, _po, _bl in rig.step(0.0, v_ego=30.0, n=600)]    # demand gone
    ev = brake_events(trace)
    assert len(ev) == 1, f"expected one committed bite, got {len(ev)}: {ev}"
    s, e, peak = ev[0]
    dur = (e - s) * DT_CTRL
    assert dur >= P.NIDEC_BRAKE_MIN_DWELL - 2 * DT_CTRL, f"bite lasted {dur:.3f} s"
    assert peak >= MIN_COUNTS

  def test_short_transient_is_a_flicker_without_the_dwell(self):
    rig = Rig(NIDEC_GRADE_CREDIT_CLAMP=False, NIDEC_FRIC_MIN_AMP=0.0, NIDEC_BRAKE_MIN_DWELL=0.0)
    trace = [ab for ab, _po, _bl in rig.step(0.0, v_ego=30.0, n=50)]
    trace += [ab for ab, _po, _bl in rig.step(-0.80, v_ego=30.0, n=10)]
    trace += [ab for ab, _po, _bl in rig.step(0.0, v_ego=30.0, n=600)]
    ev = brake_events(trace)
    assert len(ev) == 1
    s, e, _p = ev[0]
    assert (e - s) * DT_CTRL < 0.25, "baseline bite was not short -- premise broken"

  def test_no_pulse_train(self):
    """A chattering demand may not produce a train of short bites."""
    rig = Rig()
    trace = []
    for _ in range(12):                       # 12 chatter cycles over 4.8 s
      trace += [ab for ab, _po, _bl in rig.step(-0.80, v_ego=30.0, n=20)]
      trace += [ab for ab, _po, _bl in rig.step(+0.05, v_ego=30.0, n=20)]
    trace += [ab for ab, _po, _bl in rig.step(+0.05, v_ego=30.0, n=400)]
    ev = brake_events(trace)
    dwell = P.NIDEC_BRAKE_MIN_DWELL
    assert all((e - s) * DT_CTRL >= dwell - 2 * DT_CTRL for s, e, _p in ev), ev
    assert all(p >= MIN_COUNTS for _s, _e, p in ev), ev
    onsets = [s for s, _e, _p in ev]
    gaps = [(b - a) * DT_CTRL for a, b in zip(onsets, onsets[1:], strict=False)]
    assert all(g >= dwell for g in gaps), f"onsets closer than the dwell: {gaps}"

  def test_chatter_is_a_pulse_train_without_the_dwell(self):
    rig = Rig(NIDEC_GRADE_CREDIT_CLAMP=False, NIDEC_FRIC_MIN_AMP=0.0, NIDEC_BRAKE_MIN_DWELL=0.0)
    trace = []
    for _ in range(12):
      trace += [ab for ab, _po, _bl in rig.step(-0.80, v_ego=30.0, n=20)]
      trace += [ab for ab, _po, _bl in rig.step(+0.05, v_ego=30.0, n=20)]
    ev = brake_events(trace)
    assert len(ev) >= 6, f"baseline produced only {len(ev)} bites -- premise broken"

  def test_hold_floors_at_the_felt_floor_not_the_window_peak(self):
    """COMMIT v2.  The hold's floor is the minimum FELT amplitude, a constant; it is NOT
    the peak of the window.

    v1 latched `max(hold_target, apply_brake)` and never let the floor fall, so a 250 ms
    transient spike became a full-amplitude 2 s bite.  Here the command must track the
    demand up and RELAX BACK to MIN_COUNTS when the demand goes away.
    """
    rig = Rig()
    rig.step(0.0, v_ego=30.0, n=100)
    spike = [ab for ab, _po, _bl in rig.step(-2.50, v_ego=30.0, n=25)]     # 250 ms
    after = [ab for ab, _po, _bl in rig.step(0.0, v_ego=30.0, n=400)]
    assert max(spike) > 3 * MIN_COUNTS, f"spike only reached {max(spike)} counts"
    # the tail is held, but only at the felt floor -- never at the peak
    tail = after[20:]                     # past the demand path's own rate-limited release
    assert max(tail) == MIN_COUNTS, f"tail held {max(tail)} counts, floor is {MIN_COUNTS}"
    assert min(tail[:100]) == MIN_COUNTS, "the floor did not hold"
    # ...and the whole application still satisfies both floors
    ev = brake_events(spike + after)
    assert len(ev) == 1, ev
    s, e, peak = ev[0]
    assert peak >= MIN_COUNTS
    assert (e - s) * DT_CTRL >= P.NIDEC_BRAKE_MIN_DWELL - 2 * DT_CTRL

  def test_hold_never_exceeds_the_live_demand_by_more_than_the_felt_floor(self):
    """CEILING, swept over the full authorised band.  The whole point of v2: over-delivery
    relative to what the friction path is actually asking for must itself be below what
    Adam can feel.  v1's excess reached 256 counts (2.78 m/s^2).

    ⚠ One rig per speed, WALKING every depth inside it: each `CarController` costs three
    `Params()` handles and a daemon thread (`HondaParamWriter`), so a fresh rig per grid
    point exhausts the process.  Walking is also the stronger test -- state carries over.
    """
    worst = 0
    for v in (8.0, 20.0, 34.0):
      rig = Rig()
      for ask in np.arange(-4.0, -0.05, 0.10):
        rig.step(+0.60, v_ego=v, n=60)              # settle, and release any held bite
        rows = rig.step(float(ask), v_ego=v, n=120) + rig.step(0.0, v_ego=v, n=260)
        for ab, _po, bl in rows:
          demand = int(np.clip(bl * P.NIDEC_BRAKE_MAX, 0, P.NIDEC_BRAKE_MAX - 1))
          worst = max(worst, ab - demand)
    assert worst <= MIN_COUNTS, f"command exceeded the live demand by {worst} counts"

  def test_commit_never_steps_the_wire_up(self):
    """The floor may only stop the command falling.  A commit that ADVANCES the floor on
    its own frame stepped the wire up 24 counts (0.26 m/s^2) at every single brake onset --
    found by the adversarial envelope, invisible to every floor-only invariant."""
    limit = round(3.0 * (2 * DT_CTRL) * P.NIDEC_BRAKE_MAX) + 1   # the demand path's own rate
    off = dict(NIDEC_GRADE_CREDIT_CLAMP=False, NIDEC_FRIC_MIN_AMP=0.0, NIDEC_BRAKE_MIN_DWELL=0.0)
    worst = worst_base = 0
    for v in (8.0, 20.0, 34.0):
      # settle at ask 0 (NOT positive): a positive ask drives pcm_off > 0, which arms the
      # PRE-EXISTING `last_pcm_off > 0` guardrail, and that gate releases a fully ramped
      # brake in a single 253-count step in BOTH arms.  That defect is real and on the
      # record, but it is not this one, so it is excluded by construction here and pinned
      # by the ship-vs-baseline comparison below instead.
      rig, base = Rig(), Rig(**off)
      for ask in np.arange(-4.0, -0.05, 0.10):
        for r in (rig, base):
          r.step(0.0, v_ego=v, n=300)               # > dwell, so each depth commits fresh
          r.step(float(ask), v_ego=v, n=120)
      worst = max(worst, int(np.diff(rig.wire_trace()[:, 0]).max()))
      worst_base = max(worst_base, int(np.diff(base.wire_trace()[:, 0]).max()))
    assert worst <= limit, f"wire stepped up {worst} counts in one frame (limit {limit})"
    assert worst <= worst_base, f"the hold added an up-step: {worst} vs baseline {worst_base}"

  def test_dwell_does_not_survive_disengagement(self):
    rig = Rig()
    rig.step(-0.80, v_ego=30.0, n=20)
    assert rig.cc.brake_hold_frames > 0
    trace = [ab for ab, _po, _bl in rig.step(0.0, v_ego=30.0, long_active=False, n=10)]
    assert max(trace) == 0, trace
    assert rig.cc.brake_hold_frames == 0
    assert rig.cc.brake_hold_ramp == 0.0

  def test_invariant_holds_under_the_low_gear_lift_guard(self):
    """The highest-traffic path on the hilly leg this fix targets.

    Under the lift guard `eb_credit` is zeroed and NIDEC_LIFT_HONEST_FRICTION replaces
    `fric_demand` with the RAW ask (no aero/engine-brake credit), so the demand is
    systematically larger and MIN_AMP passes far more often -- and the MIN_AMP hunk sits
    immediately after that branch. Gear ratio 28 rpm/kph is above NIDEC_LIFT_RATIO_ON.
    """
    guard = dict(engine_rpm=2800.0, xmission_speed=100.0)
    bad = []
    for ask in np.arange(-1.20, -0.005, 0.02):
      for pitch in (0.0, 0.04, -0.03):
        rig = Rig()
        rig.step(-0.05, v_ego=26.0, n=30, **guard)      # latch the guard on
        assert rig.cc.lift_guard
        trace = [ab for ab, _po, _bl in rig.step(float(ask), v_ego=26.0, pitch=pitch,
                                                 n=250, **guard)]
        for s, e, peak in brake_events(trace):
          if peak < MIN_COUNTS or (e < len(trace) and (e - s) * DT_CTRL
                                   < P.NIDEC_BRAKE_MIN_DWELL - 2 * DT_CTRL):
            bad.append((round(float(ask), 3), pitch, peak, round((e - s) * DT_CTRL, 3)))
    assert not bad, f"invariant violated under the lift guard: {bad[:10]}"

  def test_dwell_never_arms_on_other_nidec_hondas(self):
    rig = Rig(fingerprint=CAR.HONDA_CIVIC)
    rig.step(-0.80, v_ego=30.0, n=40)
    assert rig.cc.brake_hold_frames == 0
    assert rig.cc.brake_hold_ramp == 0.0


# --------------------------------------------------------------------------------------
# CEILINGS -- the side of every invariant that nobody tested before 2026-07-25
# --------------------------------------------------------------------------------------

class TestWireCeiling:
  def test_panda_arithmetic_is_right(self):
    """Pin the reimplementation against panda's own numbers before trusting it.

    255 is legal, 256 is not, and 256 is exactly what v1's `hold_target` produced: it
    clipped to NIDEC_BRAKE_MAX where every other path clips to NIDEC_BRAKE_MAX - 1.
    """
    assert panda_decode_brake(bytes([0x3F, 0xC0, 0, 0, 0, 0, 0, 0])) == 255
    assert panda_decode_brake(bytes([0x40, 0x01, 0, 0, 0, 0, 0, 0])) == 256
    assert panda_brake_tx(bytes([0x3F, 0xC0, 0, 0, 0, 0, 0, 0]))[0] is True
    assert panda_brake_tx(bytes([0x40, 0x01, 0, 0, 0, 0, 0, 0]))[0] is False
    # and the frame is rejected WHOLE -- brake lights and cancel bit go with it
    assert panda_brake_tx(bytes([0x00, 0x00, 0, 0, 0, 0, 0, 0]), controls_allowed=False)[0] is True
    assert panda_brake_tx(bytes([0x01, 0x00, 0, 0, 0, 0, 0, 0]), controls_allowed=False)[0] is False

  def test_params_pin_the_wire_limits(self):
    assert P.NIDEC_BRAKE_MAX - 1 == PANDA_MAX_BRAKE
    assert P.NIDEC_GAS_MAX == PANDA_MAX_GAS      # zero margin, by design -- see values.py

  @pytest.mark.parametrize("bad", [{"NIDEC_BRAKE_MAX": 257}, {"NIDEC_GAS_MAX": 199}])
  def test_wire_limit_asserts_fire(self, bad):
    CP = interfaces[CAR.HONDA_ODYSSEY].get_non_essential_params(CAR.HONDA_ODYSSEY)
    Bad = type("BadParams", (CarControllerParams,), bad)
    with pytest.raises(AssertionError):
      Bad(CP)

  @pytest.mark.parametrize("knobs", [{}, {"NIDEC_GRADE_CREDIT_CLAMP": False,
                                          "NIDEC_FRIC_MIN_AMP": 0.0,
                                          "NIDEC_BRAKE_MIN_DWELL": 0.0}],
                           ids=["ship", "revert"])
  def test_wire_is_never_rejected_over_the_full_authorised_band(self, knobs):
    """The acceptance gate for the 2026-07-25 blocker, on the PACKED frame.

    Swept to NIDEC_ACCEL_MIN = -4.0 (not the -1.45 the corpus reaches) across the grade
    and speed range, plus the low-gear lift guard, because the saturation frontier is
    jointly minimised at low speed on a steep downgrade (-1.73 m/s^2 at 5 m/s on -10%),
    1.2 m/s^2 shallower than the flat number anyone probed separately.
    """
    bad = []
    for pitch in (-0.10, -0.05, 0.0, 0.05):
      for v, guard in ((5.0, False), (20.0, False), (35.0, False), (20.0, True)):
        rig = Rig(**knobs)
        g = dict(engine_rpm=2800.0, xmission_speed=100.0) if guard else {}
        for ask in np.arange(-4.0, -0.4, 0.10):     # one rig per condition, walk the depths
          rig.step(-0.05, v_ego=v, pitch=pitch, n=40, **g)
          rig.step(float(ask), v_ego=v, pitch=pitch, n=250, **g)
        w = rig.wire_trace()
        sent = w[:, 4] > 0
        if w[:, 0].max() > PANDA_MAX_BRAKE or (w[sent, 1] < 1).any() or (w[:, 3] < 1).any():
          bad.append((pitch, v, guard, int(w[:, 0].max()), int((w[sent, 1] < 1).sum())))
        assert w[:, 0].max() >= 200, (pitch, v, guard, w[:, 0].max())   # depth really reached
    assert not bad, f"{len(bad)} rejected/over-limit configurations, first: {bad[:5]}"

  def test_the_v1_hold_target_arithmetic_would_have_saturated(self):
    """Not vacuous: the exact expression that shipped this morning reaches 256, so the
    sweep above is testing a real boundary and not an unreachable one."""
    fric = (1.0 - P.NIDEC_MODEL_BRAKE_KNEE) * P.NIDEC_MODEL_BRAKE_PLANT
    v1 = int(np.clip(fric / P.NIDEC_MODEL_BRAKE_PLANT + P.NIDEC_MODEL_BRAKE_KNEE, 0.0, 1.0)
             * P.NIDEC_BRAKE_MAX)
    assert v1 == P.NIDEC_BRAKE_MAX == 256
    assert fric == pytest.approx(2.639, abs=1e-3)     # ask ~ -2.94 flat @ 30 m/s
    assert panda_brake_tx(bytes([v1 >> 2, (v1 & 3) << 6, 0, 0, 0, 0, 0, 0]))[0] is False
    # ...and the floor v2 uses instead is three bits clear of the ceiling
    assert MIN_COUNTS < PANDA_MAX_BRAKE // 4

  def test_packer_consistency(self):
    """The value the controller passes to `create_brake_command` must be the value panda
    decodes.  A silent scale change in the DBC is the same class of bug as the 256."""
    rig = Rig()
    for ask in (-0.5, -1.5, -3.0, -4.0, 0.0):
      rig.step(ask, v_ego=25.0, n=80)
    w = rig.wire_trace()
    sent = w[:, 4] > 0
    assert sent.any()
    # apply_brake_last is exactly what was handed to the packer on the last 50 Hz frame
    assert int(w[:, 0].max()) <= PANDA_MAX_BRAKE
    assert (w[:, 2] <= PANDA_MAX_GAS).all()

  def test_gas_side_is_within_panda_limit_while_active(self):
    """M4: NIDEC_GAS_MAX == panda max_gas exactly and is commanded on every active frame."""
    rig = Rig()
    rig.step(1.2, v_ego=20.0, n=200)
    w = rig.wire_trace()
    assert w[:, 2].max() == PANDA_MAX_GAS, w[:, 2].max()
    assert (w[:, 3] == 1).all()


# --------------------------------------------------------------------------------------
# the committed bite's releases
# --------------------------------------------------------------------------------------

class TestHoldReleases:
  def _bite_then(self, ask_after, n_after=400, **knobs):
    rig = Rig(**knobs)
    rig.step(0.0, v_ego=30.0, n=100)
    tr = [ab for ab, _po, _bl in rig.step(-0.80, v_ego=30.0, n=20)]
    tr += [ab for ab, _po, _bl in rig.step(ask_after, v_ego=30.0, n=n_after)]
    return brake_events(tr), rig

  def test_meaningful_positive_ask_releases_the_hold(self):
    ev, rig = self._bite_then(+0.50)
    assert len(ev) == 1
    dur = (ev[0][1] - ev[0][0]) * DT_CTRL
    assert dur < P.NIDEC_BRAKE_MIN_DWELL, f"hold survived a +0.50 ask for {dur:.2f} s"
    assert dur >= P.NIDEC_BRAKE_HOLD_RELEASE_T - 4 * DT_CTRL, dur
    assert rig.cc.brake_hold_frames == 0

  def test_release_needs_the_debounce(self):
    """A positive blip shorter than RELEASE_T must NOT break the bite."""
    rig = Rig()
    rig.step(0.0, v_ego=30.0, n=100)
    tr = [ab for ab, _po, _bl in rig.step(-0.80, v_ego=30.0, n=20)]
    tr += [ab for ab, _po, _bl in rig.step(+0.50, v_ego=30.0, n=20)]    # 0.20 s < 0.35 s
    tr += [ab for ab, _po, _bl in rig.step(0.00, v_ego=30.0, n=400)]
    ev = brake_events(tr)
    assert len(ev) == 1
    assert (ev[0][1] - ev[0][0]) * DT_CTRL >= P.NIDEC_BRAKE_MIN_DWELL - 2 * DT_CTRL, ev

  @pytest.mark.parametrize("ask", [0.0, 0.05, 0.086, 0.15])
  def test_release_never_fires_below_the_measured_overhang(self, ask):
    """The published rejections of an `actuators.accel > 0` release were measured on
    traces whose entire dwell overhang sat at an ask of at most +0.086 m/s^2.  Those
    measurements must still hold: the release threshold is strictly above them."""
    assert ask < P.NIDEC_BRAKE_HOLD_RELEASE_A
    ev, _rig = self._bite_then(ask)
    assert len(ev) == 1
    assert (ev[0][1] - ev[0][0]) * DT_CTRL >= P.NIDEC_BRAKE_MIN_DWELL - 2 * DT_CTRL, ev

  def test_descent_latch_releases_the_hold(self):
    """The descent latch forces `friction_ms2 = 0` because the PCM downshift serves the
    band; a bite held through it inverts the feature.  Measured on the adversarial
    envelope, v1 commanded 152 counts (1.51 m/s^2) for the full 2.00 s in-latch."""
    rig = Rig()
    # a bite, then conditions that latch descent (over set speed, downhill, mild ask)
    # v_err = vEgo - setSpeed must land in (BAND_LOW, BAND_TOP) = (-0.5, +0.83) m/s
    rig.step(-0.80, v_ego=25.0, pitch=0.0, set_speed=32.0, n=30)
    assert rig.cc.brake_hold_frames > 0
    tr = []
    for _ in range(400):
      ab, _po, _bl = rig.step(-0.02, v_ego=25.0, pitch=-0.05, set_speed=24.5)
      tr.append((ab, bool(rig.cc.descent)))
    latched = [i for i, (_ab, d) in enumerate(tr) if d]
    assert latched, "descent never latched -- test is blind"
    over = [i for i, (ab, d) in enumerate(tr) if d and ab > KNEE_COUNTS]
    # the latch is evaluated at 100 Hz and 0x1FA is packed at 50 Hz, so the wire can carry
    # the previous command for at most one 50 Hz period past the latch edge
    assert all(i <= latched[0] + 1 for i in over), (latched[0], over[:5])
    assert len(over) <= 2, over[:5]

  def test_gas_override_releases_the_hold(self):
    """panda blocks 0x1FA the moment the pedal goes down (`get_longitudinal_allowed`), so
    a held bite there is a dropped frame, not a brake."""
    rig = Rig()
    rig.step(-0.80, v_ego=30.0, n=20)
    assert rig.cc.brake_hold_frames > 0
    rig.step(0.30, v_ego=30.0, long_active=False, gas_pressed=True, n=20)
    assert rig.cc.brake_hold_frames == 0

  def test_full_disengage_resets_the_hold_state(self):
    """`enabled=False` as well as `longActive=False`.  The 50 Hz block is gated only on
    `self.CP.openpilotLongitudinalControl` (a static CarParam), so it keeps running while
    disengaged and the reset is unconditional -- but if that gate ever became dynamic the
    countdown would FREEZE instead of resetting and a stale floor would re-appear on the
    next engage, before any demand exists.  Pinned here."""
    rig = Rig()
    rig.step(-0.80, v_ego=30.0, n=20)
    assert rig.cc.brake_hold_frames > 0 and rig.cc.brake_hold_ramp > 0.0
    rig.step(0.0, v_ego=30.0, enabled=False, long_active=False, n=10)
    assert rig.cc.brake_hold_frames == 0 and rig.cc.brake_hold_ramp == 0.0
    trace = [ab for ab, _po, _bl in rig.step(0.0, v_ego=30.0, n=20)]   # re-engage, no demand
    assert max(trace) == 0, trace

  def test_gas_override_frame_drops_are_bounded_and_pre_existing(self):
    """The `!get_longitudinal_allowed` arm of panda's tx hook, driven by a REAL controller
    trace rather than synthetic bytes.

    DOCUMENTED, NOT FIXED: openpilot needs a frame or two to zero the command after the
    pedal goes down, so 0x1FA is dropped for a few frames.  It is bit-identical to the
    pre-change code (the hold is released by `not CC.longActive` on the same frame), and
    fixing it means touching the shipped gas-override handoff.  Bounded here so it cannot
    grow, and so the drop is always THIS cause and never an over-limit command.
    """
    off = dict(NIDEC_GRADE_CREDIT_CLAMP=False, NIDEC_FRIC_MIN_AMP=0.0, NIDEC_BRAKE_MIN_DWELL=0.0)
    drops = {}
    for arm, knobs in (("ship", {}), ("revert", off)):
      rig = Rig(**knobs)
      rig.step(-1.50, v_ego=30.0, n=60)                                   # a deep bite
      rig.step(+0.20, v_ego=30.0, long_active=False, gas_pressed=True, n=60)   # pedal down
      rig.step(0.0, v_ego=30.0, n=60)
      w = rig.wire_trace()
      sent = w[:, 4] > 0
      bad = sent & (w[:, 1] < 1)
      drops[arm] = int(bad.sum())
      assert w[:, 0].max() <= PANDA_MAX_BRAKE       # never the over-limit cause
    assert drops["ship"] <= max(3, drops["revert"]), drops
    assert drops["ship"] == drops["revert"], drops   # pre-existing, unchanged by this diff


# --------------------------------------------------------------------------------------
# the grade credit clamp
# --------------------------------------------------------------------------------------

class TestGradeCreditClamp:
  def test_uphill_decel_ask_never_commands_gas(self):
    """On a decel ask the grade FF may not push the pcm_off COMMAND positive.

    This is the 2026-07-24 quartet-drive root cause: uphill the credit inverted a decel
    ask into a gas request, which grade-activated the `last_pcm_off > 0` friction
    guardrail and zeroed the brake exactly where the ask needed serving.
    """
    inverted_without_clamp = 0
    for pitch in (0.01, 0.02, 0.03, 0.05, 0.08):
      for ask in (-0.10, -0.20, -0.35, -0.60):
        ship, base = Rig(), Rig(NIDEC_GRADE_CREDIT_CLAMP=False)
        po_ship = ship.step(ask, v_ego=28.0, pitch=pitch, n=150)[-1][1]
        po_base = base.step(ask, v_ego=28.0, pitch=pitch, n=150)[-1][1]
        assert po_ship <= 1e-9, f"pitch {pitch} ask {ask}: pcm_off {po_ship:+.3f} > 0"
        assert po_base >= po_ship - 1e-9      # the clamp can only reduce the command
        inverted_without_clamp += int(po_base > 1e-9)
    assert inverted_without_clamp > 0, "baseline never inverted a decel ask -- test is blind"

  def test_clamp_is_inert_on_non_decel_asks(self):
    """`actuators.accel < 0.0` gates it: nothing changes when the ask is not a decel."""
    ship, base = Rig(), Rig(NIDEC_GRADE_CREDIT_CLAMP=False)
    for ask in (0.0, 0.05, 0.30, 0.90):
      for pitch in (-0.05, 0.0, 0.05):
        assert ship.step(ask, v_ego=28.0, pitch=pitch, n=60)[-1] == \
               base.step(ask, v_ego=28.0, pitch=pitch, n=60)[-1], (ask, pitch)

  def test_clamp_is_command_side_only(self):
    """The friction demand is unchanged by construction: where the unclamped credit would
    have pushed a_des_b positive, fric_demand was already 0 and stays 0.

    Compared on `brake_last` (the pre-guardrail brake fraction), which is exactly the
    friction demand path's output and updates every frame.
    """
    ship = Rig(NIDEC_FRIC_MIN_AMP=0.0, NIDEC_BRAKE_MIN_DWELL=0.0)
    base = Rig(NIDEC_GRADE_CREDIT_CLAMP=False, NIDEC_FRIC_MIN_AMP=0.0, NIDEC_BRAKE_MIN_DWELL=0.0)
    rng = np.random.default_rng(7)
    for _ in range(600):
      ask = float(rng.uniform(-1.0, -0.01))
      pitch = float(rng.uniform(-0.08, 0.08))
      v = float(rng.uniform(15.0, 35.0))
      _a_ab, _a_po, a_bl = ship.step(ask, v_ego=v, pitch=pitch)
      _b_ab, _b_po, b_bl = base.step(ask, v_ego=v, pitch=pitch)
      assert a_bl == b_bl, (ask, pitch, v, a_bl, b_bl)


# --------------------------------------------------------------------------------------
# staging lock + documented gaps
# --------------------------------------------------------------------------------------

class TestStagingLock:
  def test_min_amp_without_dwell_is_rejected(self):
    """`clamp_amp` -- MIN_AMP without the dwell -- is the one arm measured WORSE than the
    shipping code on both validation drives.  It must not be buildable."""
    CP = interfaces[CAR.HONDA_ODYSSEY].get_non_essential_params(CAR.HONDA_ODYSSEY)

    class BadParams(CarControllerParams):
      NIDEC_FRIC_MIN_AMP = 0.20
      NIDEC_BRAKE_MIN_DWELL = 0.0

    with pytest.raises(AssertionError):
      BadParams(CP)

    class OkParams(CarControllerParams):
      NIDEC_FRIC_MIN_AMP = 0.0
      NIDEC_BRAKE_MIN_DWELL = 0.0

    OkParams(CP)                       # full revert is fine
    CarControllerParams(CP)            # and so are the shipping values

  def test_dwell_without_min_amp_degenerates_to_the_knee(self):
    """DOCUMENTED DEGENERATE CONFIG, pinned so it cannot drift silently.

    The staging lock forbids MIN_AMP without the dwell; the reverse (dwell alone) IS
    buildable and is a plausible A/B.  There `hold_floor` collapses to `knee_counts` --
    the hold keeps the pads at the hydraulic preload for the dwell, which is brake lights
    and an armed VSA pump for ZERO decel, and is not even an "application" by the
    `> KNEE_COUNTS` convention every invariant in this file uses.  If someone wants to A/B
    the dwell, A/B it WITH MIN_AMP; the dwell's semantic is "hold the felt floor", and
    without a felt floor there is nothing meaningful to hold.
    """
    rig = Rig(NIDEC_FRIC_MIN_AMP=0.0)
    rig.step(0.0, v_ego=30.0, n=50)
    trace = [ab for ab, _po, _bl in rig.step(-0.80, v_ego=30.0, n=10)]
    trace += [ab for ab, _po, _bl in rig.step(0.0, v_ego=30.0, n=400)]
    assert set(trace[20:190]) == {KNEE_COUNTS}, sorted(set(trace[20:190]))
    assert counts_to_ms2(KNEE_COUNTS) == 0.0        # ...which delivers nothing
    # brake lights stay on for the whole dwell, but the floor invariants see only the
    # demand's own 0.08 s bite -- the 1.9 s of preload is invisible to every one of them
    on_s = sum(1 for x in trace if x > 0) * DT_CTRL
    assert on_s >= P.NIDEC_BRAKE_MIN_DWELL - 0.1, on_s
    ev = brake_events(trace)
    assert len(ev) == 1 and (ev[0][1] - ev[0][0]) * DT_CTRL < 0.2, ev

  def test_shipping_values_are_the_scored_ones(self):
    assert P.NIDEC_GRADE_CREDIT_CLAMP is True
    assert P.NIDEC_FRIC_MIN_AMP == 0.20
    assert P.NIDEC_BRAKE_MIN_DWELL == 2.0
    assert P.NIDEC_BRAKE_HOLD_RELEASE_A == 0.20
    assert P.NIDEC_BRAKE_HOLD_RELEASE_T == 0.35
    # the release threshold IS the felt floor: brake and gas can then never disagree by
    # more than one felt unit in either direction
    assert P.NIDEC_BRAKE_HOLD_RELEASE_A == P.NIDEC_FRIC_MIN_AMP
    # the commit knee literal must equal the map's own knee (M6)
    assert round(P.NIDEC_MODEL_BRAKE_KNEE * P.NIDEC_BRAKE_MAX) == KNEE_COUNTS == 13
    # the servo-ID dither ships on the branch but must stay inert
    assert P.NIDEC_DITHER_AMP == 0.0


class TestLowSpeedGap:
  def test_creep_brake_cannot_commit_a_dwell(self):
    """The commit is gated on `friction_ms2 > 0`, so the legacy `creep_brake` stop-hold
    ramp -- which crosses the 13-count knee with ZERO friction demand below 2.3 m/s -- can
    no longer commit a bite.  Without that gate a launch inside the creep window would
    drag the brake for the whole dwell."""
    rig = Rig()
    trace = [ab for ab, _po, _bl in rig.step(0.0, v_ego=0.5, n=40)]
    assert max(trace) > KNEE_COUNTS, "creep_brake did not cross the knee -- test is blind"
    assert rig.cc.brake_hold_frames == 0, "creep_brake committed a dwell"
    # and the creep command is bounded by the legacy ramp itself
    creep_counts = int(np.clip((2.3 - 0.5) / 2.3 * 0.15, 0.0, 1.0) * P.NIDEC_BRAKE_MAX)
    assert max(trace) <= creep_counts + 1, (max(trace), creep_counts)

  @pytest.mark.parametrize("v", [0.2, 0.5, 1.0])
  def test_creep_command_is_NOT_covered_by_the_felt_floor(self, v):
    """DOCUMENTED SCOPE, pinned so the comment and the code can never drift.

    `NIDEC_FRIC_MIN_AMP` floors the DEMAND; below 2.3 m/s there is no demand, so the creep
    ramp's own 19-32 count command (0.065-0.206 m/s^2) sits below the felt floor and is
    bit-identical to the pre-change code.  The "no application below the felt floor" claim is
    scoped to applications whose amplitude comes from `friction_ms2`.
    Measured unreachable while engaged on this car (0 engaged frames < 2.3 m/s over 130
    routes / 119.3 engaged min, min engaged vEgo 4.97 m/s) -- but the one route known to
    contain openpilot-controlled full stops is not in the local cache, so that is not proof.
    """
    ship = [ab for ab, _po, _bl in Rig().step(0.0, v_ego=v, n=40)]
    base = [ab for ab, _po, _bl in Rig(NIDEC_GRADE_CREDIT_CLAMP=False, NIDEC_FRIC_MIN_AMP=0.0,
                                       NIDEC_BRAKE_MIN_DWELL=0.0).step(0.0, v_ego=v, n=40)]
    assert ship == base, "creep regime is NOT bit-identical to the pre-change code"
    # bounded by the legacy creep ramp, and (except right at a standstill) below the floor
    creep_counts = int(np.clip((2.3 - v) / 2.3 * 0.15, 0.0, 1.0) * P.NIDEC_BRAKE_MAX)
    assert KNEE_COUNTS < max(ship) <= creep_counts + 1, (v, max(ship), creep_counts)
    assert (max(ship) < MIN_COUNTS) == (v >= 0.5), (v, max(ship), MIN_COUNTS)

  def test_no_creep_commit_above_creep_speed(self):
    """Above 2.3 m/s `creep_brake` is identically zero, so a commit requires real demand."""
    rig = Rig()
    trace = [ab for ab, _po, _bl in rig.step(0.0, v_ego=2.4, n=40)]
    assert max(trace) == 0
    assert rig.cc.brake_hold_frames == 0
