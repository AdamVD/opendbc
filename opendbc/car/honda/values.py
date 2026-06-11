from dataclasses import dataclass, field
from enum import Enum, IntFlag

from opendbc.car import Bus, CarSpecs, DbcDict, PlatformConfig, Platforms, structs, uds
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.docs_definitions import CarFootnote, CarHarness, CarDocs, CarParts, Column, SupportType
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries, p16

Ecu = structs.CarParams.Ecu
VisualAlert = structs.CarControl.HUDControl.VisualAlert
GearShifter = structs.CarState.GearShifter


class CarControllerParams:
  # Allow small margin below -3.5 m/s^2 from ISO 15622:2018 since we
  # perform the closed loop control, and might need some
  # to apply some more braking if we're on a downhill slope.
  # Our controller should still keep the 2 second average above
  # -3.5 m/s^2 as per planner limits
  NIDEC_ACCEL_MIN = -4.0  # m/s^2
  NIDEC_ACCEL_MAX = 1.6  # m/s^2, lower than 2.0 m/s^2 for tuning reasons

  NIDEC_ACCEL_LOOKUP_BP = [-1., 0., .6]
  NIDEC_ACCEL_LOOKUP_V = [-4.8, 0., 2.0]

  NIDEC_MAX_ACCEL_V = [0.5, 2.4, 1.4, 0.6]
  NIDEC_MAX_ACCEL_BP = [0.0, 4.0, 10., 20.]

  NIDEC_GAS_MAX = 198  # 0xc6
  NIDEC_BRAKE_MAX = 1024 // 4
  # Brake-side scale on the wind_brake (aero-drag) offset. The generic wind_brake (~0.10-0.15)
  # overstates the 2018 Odyssey's true coastdown (measured from rlog: aego -0.26 m/s2 @50mph,
  # -0.37 @65mph -> brake-fraction 0.054-0.077, i.e. ~1.5-1.8x smaller than wind_brake). A heavy
  # but aerodynamic minivan coasts down LESS than the blank-average profile (low Cd*A / high mass),
  # so the generic offset holds the friction brake off too long, leaving a dead-band between the
  # gas-side ACC saturation (~-0.33 m/s2) and brake onset (~-0.46). Scaling the brake-side offset
  # to match measured coast moves brake onset to ~-0.28, closing that dead-band ("corner too hot")
  # and improving downhill decel delivery (+~88% brake) which also reduces integrator-windup droop.
  # Gas side is unchanged; the pcm_off>0 guardrail keeps it conflict-safe. Lower = more brake.
  # (Superseded on the NIDEC_ALT path by the honest Plant-B model below; kept for other Nidecs.)
  NIDEC_BRAKE_WIND_FACTOR = 0.6

  # ---- Plant-B: passive/friction realm honest model (FINDINGS_channel_plants_2026-06-09) ----
  # The PCM servo realm (pcm_off) and the friction realm are DIFFERENT PLANTS. Wire-level rlog
  # decomposition of the downhill droop event (deficit +0.95 m/s^2) split exactly into: brake map
  # optimism (compute_gas_brake assumes full brake = 4.8 m/s^2, measured 3.07) + aero over-credit
  # (unitless wind_brake ~0.68 m/s^2-equiv @33 m/s vs true coastdown 0.30) + grade under-comp
  # (2.2 vs 9.81 -- the servo's grade rejection DIES when the throttle closes). Same error mirrored
  # uphill: mild-decel lift over-decelerates (gravity credited at 2.2, acts at 9.81).
  NIDEC_MODEL_BRAKE_PLANT = 2.78   # m/s^2 decel at full apply_brake. Re-measured 2026-06-10 from
                                   # 2376 steady grade-corrected samples (quintet+clevpack rlogs):
                                   # 0.01086 m/s^2 per COMPUTER_BRAKE count x NIDEC_BRAKE_MAX=256,
                                   # with a ~13-count dead knee (see NIDEC_MODEL_BRAKE_KNEE).
  NIDEC_MODEL_BRAKE_KNEE = 13.0 / 256.0  # apply_brake fraction that produces no decel (hydraulic
                                   # preload/pad clearance; LSQ knee 12.6 counts). Added as a bias
                                   # whenever any friction is demanded, linearizing the response --
                                   # without it, light braking (15-30 counts) delivered 0.07 m/s^2
                                   # where the map expected 0.25 (descent overspeed, stoplight under-
                                   # delivery +0.4-0.7).
  NIDEC_MODEL_ENGINE_BRAKE = 0.05  # m/s^2 engine-brake credit at the pcm_off floor (throttle closed)
                                   # ON TOP of the aero/coastdown table. Re-measured 2026-06-10: total
                                   # passive decel (grade-corrected, no friction) is 0.21-0.32 m/s^2
                                   # across 18-36 m/s ~= the coastdown table + only ~0.05; the old
                                   # 0.30 double-counted engine braking already baked into the
                                   # measured-coastdown wind_ms2 -> every moderate decel was commanded
                                   # ~0.25 m/s^2 light (P1 downhill deficit +0.27, hill-bottom swing).
  NIDEC_MODEL_GRADE_BLEND_LO = 0.05  # |decel request| where the grade-coefficient blend starts
  NIDEC_MODEL_GRADE_BLEND_HI = 0.35  # ... and where it reaches full 9.81 (passive realm)

  # Model-based longitudinal feedforward for NIDEC_ALT_PCM_ACCEL (Odyssey)
  # Plant: aego = K(v) * pcm_off - g*sin(pitch),  K(v) = K0 - K1*v
  # Identified from ~302 min of 2018 Odyssey logs (install e3652ba0)
  NIDEC_MODEL_K0 = 0.178        # plant gain at 0 m/s (1/s)
  NIDEC_MODEL_K1 = 0.0022       # velocity coefficient (1/s per m/s)
  NIDEC_MODEL_K_MIN = 0.08      # lower clamp on K(v)
  NIDEC_MODEL_K_MAX = 0.20      # upper clamp on K(v)
  NIDEC_MODEL_PCM_OFF_MAX = 8.0  # max pcm_speed offset above vEgo (m/s). Bump-test value (Tier-2,
                                 # EXTRAPOLATED beyond logged data which only covered pcm_off<=5): from
                                 # the tuning dashboard a constant pcm_off=+5 (the prior cap) delivered
                                 # only ~0.65 m/s2 and tracked parallel to vEgo 50->74 mph, confirming the
                                 # CAP (not gain) is the dominant accel limiter and plant K is ~speed-flat.
                                 # Deliverable_max = K*5 ~= 0.5-0.65, far below the ~1.6 target. At 8.0
                                 # deliverable ~= K*8 ~= 0.8-1.0. The sustained +8 rail (cmd not feedback-
                                 # chosen) also serves as a clean rail-K read for the next ship decision.
  NIDEC_MODEL_PCM_OFF_MIN = -1.5 # floor for pcm_speed offset below vEgo (m/s). At 0.0, Honda ACC
                                 # sat at pcm_speed=vEgo and applied throttle to hold speed while the
                                 # direct brake channel fought it — confirmed from 2026-06-03 drive:
                                 # 5311/5311 braking samples were gas+brake conflicts (33% of active
                                 # time), including 2591 on flat ground behind a lead car, max brake
                                 # 0.40. At -1.5, pcm_speed trails vEgo during decel so Honda ACC
                                 # idles/engine-brakes instead of fighting. Recovery to gas: 0.25s
                                 # (vs 1.3s at the deployed -8.0). Watch: ~0.25s pickup hesitation
                                 # after following a lead; raise NIDEC_MODEL_RATE or reduce |MIN|
                                 # if it shows up.
  NIDEC_MODEL_DECEL_SOFT = 2.0   # decel-side gain softener (asymmetric map). The inverse-model
                                 # pcm_off=a_des/K uses the full 1/K gain (~7.7) for accel (the "oomph"),
                                 # but applied symmetrically it slams pcm_off to the PCM_OFF_MIN floor for
                                 # any ease-off past a_des~-0.2 — a relay/cliff (66% of ease-off samples
                                 # floored; confirmed NEW with the 2026-06-02 gain increase vs the old
                                 # map's graded lift-off). Honda's ACC throttle only modulates over pcm_off
                                 # [0,-1.5] (coast/engine-brake, saturating ~-0.3 m/s2; it does NOT friction
                                 # brake), so the decel side needs far less gain. Dividing the decel-side
                                 # gain by this restores a graded lift-off (reaches the floor near a_des
                                 # -0.4) and removes the relay that feeds the lead-follow oscillation, while
                                 # keeping full accel gain and the -1.5 floor. pcm_off stays <=0 on decel so
                                 # the ACC never fights the brake; hard decel still floors + brakes unchanged.
                                 # 1.0 = symmetric (old cliff); ~2 grades it; >~3 erodes decel response
                                 # margin on a slowing lead. Tune on-car.
  NIDEC_MODEL_RATE = 6.0        # pcm_off rate limit DOWNWARD (m/s per s). Governs ease-off smoothness;
                                # works with DECEL_SOFT to keep the graded lift-off (don't raise casually).
  NIDEC_MODEL_RATE_UP = 10.0    # pcm_off rate limit UPWARD (m/s per s). 2026-06-09 rlog step analysis
                                # (drive 00000027): gas response t_half scales with ask amplitude
                                # (0.4s @ +0.3 ask, 1.35s @ +0.9-1.1) = slew-limit fingerprint, and
                                # ask/(K*RATE) at RATE=6 matched the big-ask t_half exactly -- the old
                                # symmetric 6.0 ramp (1.33s for 0->8) was the binding "slow to strength"
                                # lag, not the PCM (dead time only ~0.35s). At 10, a 1.0 m/s^2 ask ramps
                                # in ~0.8s. The PCM's own internal smoothing still shapes the torque.
  # Grade feed-forward gain for the gas-side pcm_off ONLY. At the FF's operating point (small
  # pcm_off, near speed-hold) the Honda ACC's internal speed loop rejects most grade, so the
  # closed-loop grade leakage the FF must invert is only ~2.1 m/s^2, NOT full g. Measured in the
  # coast region (pcm_off~=0, brake off, both grade signs) on the 2026-06-02 drive: car does NOT
  # accelerate on -2..-3.4deg downhills (aego~=+0.003 vs +0.39 predicted by g=9.81); fit ~2.1.
  # Using full g=9.81 over-cuts gas on downhills (~3 mph droop). Brake channel + Bosch keep full g.
  # Caveat: leakage is briefly higher entering a grade and on grades steeper than ~6deg.
  NIDEC_MODEL_GRADE_G = 2.2     # m/s^2; tune on-car (raise toward 9.81 if downhill overspeed)

  # Tier-1 kickdown-surge trim (shift anticipation). The 10AT TCU announces every shift on
  # GEARBOX_AUTO.TRANS_TARGET_GEAR ~0.3-1.0s before torque transfer; under ACC the felt surge
  # peaks ~1.3s after the announcement and only G<=7 kickdowns are perceptible (med +0.6 m/s^2;
  # G8-10 med +0.1 -- 2026-06-10 corpus, 223 power-on kickdowns over 6.8h). On a power-on
  # downshift announcement, scale the positive pcm_off ask by TRIM_DEPTH and recover linearly
  # over TRIM_DECAY_S. Depth must stay well above 0 (po~=0 is the box's upshift-back trigger ->
  # hunting; also still holding the hill) and decay must finish inside the post-kickdown dwell
  # (p10 3.5s). The cut itself is shaped by NIDEC_MODEL_RATE (6/s), no cliff.
  NIDEC_TRIM_DEPTH = 0.5        # pcm_off multiplier at trigger (1.0 = feature off)
  NIDEC_TRIM_DECAY_S = 3.0      # seconds to recover multiplier from DEPTH back to 1.0
  NIDEC_TRIM_PO_MIN = 0.5       # only trim if ~1s-filtered pcm_off exceeds this (power-on context).
                                # Replay sweep (trim_replay.py, 2.28 engaged-hours): 0.5 catches 96%
                                # of felt G<=7 surges (>0.4) at 41 fires/h; 1.5 only 69% at 30/h --
                                # felt surges happen down to po~0.6, and a "false" fire just softens
                                # gas briefly on a real (if benign) downshift.
  NIDEC_TRIM_GFROM_MAX = 7      # only trim announcements stepping down FROM gear <= this
                                # (G8-10 kickdown surge med +0.1 m/s^2 -- imperceptible, leave alone)

  BOSCH_ACCEL_MIN = -3.5  # m/s^2
  BOSCH_ACCEL_MAX = 2.0  # m/s^2

  BOSCH_GAS_LOOKUP_BP = [0.0, 2.0]  # 2m/s^2
  BOSCH_GAS_LOOKUP_V = [0, 1600]

  STEER_STEP = 1  # 100 Hz
  STEER_DELTA_UP = 3  # min/max in 0.33s for all Honda
  STEER_DELTA_DOWN = 3
  STEER_GLOBAL_MIN_SPEED = 3 * CV.MPH_TO_MS

  def __init__(self, CP):
    self.STEER_MAX = CP.lateralParams.torqueBP[-1]
    # mirror of list (assuming first item is zero) for interp of signed request
    # values and verify that both arrays begin at zero
    assert CP.lateralParams.torqueBP[0] == 0
    assert CP.lateralParams.torqueV[0] == 0
    self.STEER_LOOKUP_BP = [v * -1 for v in CP.lateralParams.torqueBP][1:][::-1] + list(CP.lateralParams.torqueBP)
    self.STEER_LOOKUP_V = [v * -1 for v in CP.lateralParams.torqueV][1:][::-1] + list(CP.lateralParams.torqueV)


class HondaSafetyFlags(IntFlag):
  ALT_BRAKE = 1
  BOSCH_LONG = 2
  NIDEC_ALT = 4
  RADARLESS = 8
  BOSCH_CANFD = 16


class HondaFlags(IntFlag):
  # Detected flags
  # Bosch models with alternate set of LKAS_HUD messages
  BOSCH_EXT_HUD = 1
  BOSCH_ALT_BRAKE = 2

  # Static flags
  BOSCH = 4
  BOSCH_RADARLESS = 8

  NIDEC = 16
  NIDEC_ALT_PCM_ACCEL = 32
  NIDEC_ALT_SCM_MESSAGES = 64

  BOSCH_CANFD = 128

  HAS_ALL_DOOR_STATES = 256  # Some Hondas have all door states, others only driver door
  BOSCH_ALT_RADAR = 512
  # 1024 is available
  HYBRID = 2048
  BOSCH_TJA_CONTROL = 4096
  LKAS_MINSPEED_CUTOFF = 8192


# Car button codes
class CruiseButtons:
  RES_ACCEL = 4
  DECEL_SET = 3
  CANCEL = 2
  MAIN = 1


class CruiseSettings:
  DISTANCE = 3
  LKAS = 1


@dataclass
class HondaCarDocs(CarDocs):
  package: str = "Honda Sensing"

  def init_make(self, CP: structs.CarParams):
    if CP.flags & HondaFlags.BOSCH:
      if CP.flags & HondaFlags.BOSCH_CANFD:
        harness = CarHarness.bosch_c
      elif CP.flags & HondaFlags.BOSCH_RADARLESS:
        harness = CarHarness.bosch_b
      else:
        harness = CarHarness.bosch_a
    else:
      harness = CarHarness.nidec

    self.car_parts = CarParts.common([harness])

    if CP.carFingerprint in (CAR.HONDA_CLARITY,):
      self.car_parts = CarParts.common([CarHarness.honda_clarity])
      self.car_parts.custom_parts_url = "https://shop.retropilot.org/product/honda-clarity-proxy-board-kit"
      self.support_type: SupportType = SupportType.COMMUNITY
      self.support_link: str = "community"


class Footnote(Enum):
  CIVIC_DIESEL = CarFootnote(
    "2019 Honda Civic 1.6L Diesel Sedan does not have ALC below 12mph.",
    Column.FSR_STEERING)
  TRAFFIC_JAM_ASSIST = CarFootnote(
    "ALC is supported below 45mph only when following a lead car.",
    Column.FSR_STEERING)


@dataclass
class HondaBoschPlatformConfig(PlatformConfig):
  def init(self):
    self.flags |= HondaFlags.BOSCH


@dataclass
class HondaBoschCANFDPlatformConfig(HondaBoschPlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: 'honda_common_canfd_generated'})

  def init(self):
    super().init()
    self.flags |= HondaFlags.BOSCH_CANFD


@dataclass
class HondaNidecPlatformConfig(PlatformConfig):
  def init(self):
    self.flags |= HondaFlags.NIDEC


def radar_dbc_dict(pt_dict):
  return {Bus.pt: pt_dict, Bus.radar: 'acura_ilx_2016_nidec'}


# Certain Hondas have an extra steering sensor at the bottom of the steering rack,
# which improves controls quality as it removes the steering column torsion from feedback.
# Tire stiffness factor fictitiously lower if it includes the steering column torsion effect.
# For modeling details, see p.198-200 in "The Science of Vehicle Dynamics (2014), M. Guiggiani"


class CAR(Platforms):
  # Bosch Cars
  HONDA_NBOX_2G = HondaBoschPlatformConfig(
    [
      HondaCarDocs("Honda N-Box 2018", "All", min_steer_speed=5.),
    ],
    CarSpecs(mass=890., wheelbase=2.520, steerRatio=18.64),
    {Bus.pt: 'acura_rdx_2020_can_generated'},
  )
  HONDA_ACCORD = HondaBoschPlatformConfig(
    [
      HondaCarDocs("Honda Accord 2018-22", "All", video="https://www.youtube.com/watch?v=mrUwlj3Mi58", min_steer_speed=3. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Inspire 2018", "All", min_steer_speed=3. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Accord Hybrid 2018-22", "All", min_steer_speed=3. * CV.MPH_TO_MS),
    ],
    # steerRatio: 11.82 is spec end-to-end
    CarSpecs(mass=3279 * CV.LB_TO_KG, wheelbase=2.83, steerRatio=16.33, centerToFrontRatio=0.39, tireStiffnessFactor=0.8467),
    {Bus.pt: 'honda_civic_hatchback_ex_2017_can_generated'},
  )
  HONDA_ACCORD_11G = HondaBoschCANFDPlatformConfig(
    [
      HondaCarDocs("Honda Accord 2023-26", "All"),
      HondaCarDocs("Honda Accord Hybrid 2023-26", "All"),
  ],
    CarSpecs(mass=3477 * CV.LB_TO_KG, wheelbase=2.83, steerRatio=16.7, centerToFrontRatio=0.39),
  )
  HONDA_CIVIC_BOSCH = HondaBoschPlatformConfig(
    [
      HondaCarDocs("Honda Civic 2019-21", "All", video="https://www.youtube.com/watch?v=4Iz1Mz5LGF8",
                   footnotes=[Footnote.CIVIC_DIESEL], min_steer_speed=2. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Civic Hatchback 2017-18", min_steer_speed=12. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Civic Hatchback 2019-21", "All", min_steer_speed=12. * CV.MPH_TO_MS),
    ],
    CarSpecs(mass=1326, wheelbase=2.7, steerRatio=15.38, centerToFrontRatio=0.4),  # steerRatio: 10.93 is end-to-end spec
    {Bus.pt: 'honda_civic_hatchback_ex_2017_can_generated'},
  )
  HONDA_CIVIC_BOSCH_DIESEL = HondaBoschPlatformConfig(
    [],  # don't show in docs
    HONDA_CIVIC_BOSCH.specs,
    {Bus.pt: 'honda_civic_hatchback_ex_2017_can_generated'},
  )
  HONDA_CIVIC_2022 = HondaBoschPlatformConfig(
    [
      HondaCarDocs("Honda Civic 2022-26", "All", video="https://youtu.be/ytiOT5lcp6Q"),
      HondaCarDocs("Honda Civic Hybrid 2025-26", "All"),
      HondaCarDocs("Honda Civic Hatchback 2022-26", "All", video="https://youtu.be/ytiOT5lcp6Q"),
      HondaCarDocs("Honda Civic Hatchback Hybrid (Europe only) 2023", "All"),
      # TODO: Confirm 2024
      HondaCarDocs("Honda Civic Hatchback Hybrid 2025-26", "All"),
    ],
    HONDA_CIVIC_BOSCH.specs,
    {Bus.pt: 'honda_bosch_radarless_generated'},
    flags=HondaFlags.BOSCH_RADARLESS
  )
  HONDA_CRV_5G = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda CR-V 2017-22", min_steer_speed=15. * CV.MPH_TO_MS)],
    # steerRatio: 12.3 is spec end-to-end
    CarSpecs(mass=3410 * CV.LB_TO_KG, wheelbase=2.66, steerRatio=16.0, centerToFrontRatio=0.41, tireStiffnessFactor=0.677),
    {Bus.pt: 'honda_civic_hatchback_ex_2017_can_generated', Bus.body: 'honda_crv_ex_2017_body_generated'},
    flags=HondaFlags.BOSCH_ALT_BRAKE | HondaFlags.LKAS_MINSPEED_CUTOFF
  )
  HONDA_CRV_6G = HondaBoschCANFDPlatformConfig(
    [
      HondaCarDocs("Honda CR-V 2023-26", "All"),
      HondaCarDocs("Honda CR-V Hybrid 2023-26", "All"),
    ],
    CarSpecs(mass=1703, wheelbase=2.7, steerRatio=16.2, centerToFrontRatio=0.42),
  )
  HONDA_CRV_HYBRID = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda CR-V Hybrid 2017-22", min_steer_speed=12. * CV.MPH_TO_MS)],
    # mass: mean of 4 models in kg, steerRatio: 12.3 is spec end-to-end
    CarSpecs(mass=1667, wheelbase=2.66, steerRatio=16, centerToFrontRatio=0.41, tireStiffnessFactor=0.677),
    {Bus.pt: 'honda_civic_hatchback_ex_2017_can_generated'},
  )
  HONDA_HRV_3G = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda HR-V 2023-26", "All")],
    CarSpecs(mass=3125 * CV.LB_TO_KG, wheelbase=2.61, steerRatio=15.2, centerToFrontRatio=0.41, tireStiffnessFactor=0.5),
    {Bus.pt: 'honda_bosch_radarless_generated'},
    flags=HondaFlags.BOSCH_RADARLESS,
  )
  HONDA_CITY_7G = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda City (Brazil only) 2023", "All")],
    CarSpecs(mass=3125 * CV.LB_TO_KG, wheelbase=2.6, steerRatio=19.0, centerToFrontRatio=0.41, minSteerSpeed=23. * CV.KPH_TO_MS),
    {Bus.pt: 'honda_bosch_radarless_generated'},
    flags=HondaFlags.BOSCH_RADARLESS | HondaFlags.LKAS_MINSPEED_CUTOFF
  )
  ACURA_RDX_3G = HondaBoschPlatformConfig(
    [HondaCarDocs("Acura RDX 2019-21", "All", min_steer_speed=3. * CV.MPH_TO_MS)],
    CarSpecs(mass=4068 * CV.LB_TO_KG, wheelbase=2.75, steerRatio=11.95, centerToFrontRatio=0.41, tireStiffnessFactor=0.677),  # as spec
    {Bus.pt: 'acura_rdx_2020_can_generated'},
  )
  ACURA_RDX_3G_MMR = HondaBoschPlatformConfig(
    [HondaCarDocs("Acura RDX 2022-26", "All", min_steer_speed=70. * CV.KPH_TO_MS)],
    CarSpecs(mass=4079 * CV.LB_TO_KG, wheelbase=2.75, centerToFrontRatio=0.41, steerRatio=16.2),
    {Bus.pt: 'acura_rdx_2020_can_generated'},
    flags=HondaFlags.BOSCH_ALT_BRAKE | HondaFlags.BOSCH_ALT_RADAR,
  )
  HONDA_INSIGHT = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda Insight 2019-22", "All", min_steer_speed=3. * CV.MPH_TO_MS)],
    CarSpecs(mass=2987 * CV.LB_TO_KG, wheelbase=2.7, steerRatio=15.0, centerToFrontRatio=0.39, tireStiffnessFactor=0.82),  # as spec
    {Bus.pt: 'honda_insight_ex_2019_can_generated'},
  )
  HONDA_E = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda e 2020", "All", min_steer_speed=3. * CV.MPH_TO_MS)],
    CarSpecs(mass=3338.8 * CV.LB_TO_KG, wheelbase=2.5, centerToFrontRatio=0.5, steerRatio=16.71, tireStiffnessFactor=0.82),
    {Bus.pt: 'acura_rdx_2020_can_generated'},
  )
  HONDA_E_ADVANCE = HondaBoschPlatformConfig(
    [],  # don't show in docs, base trim already in docs
    CarSpecs(mass=1527, wheelbase=2.5, centerToFrontRatio=0.5, steerRatio=16.71, tireStiffnessFactor=0.82),
    {Bus.pt: 'honda_e_advance_2020_can_generated'}, # 8 bit LKAS_HUD in Advance trim
  )
  HONDA_PILOT_4G = HondaBoschCANFDPlatformConfig(
    [HondaCarDocs("Honda Pilot 2023-26", "All")],
    CarSpecs(mass=4660 * CV.LB_TO_KG, wheelbase=2.89, centerToFrontRatio=0.442, steerRatio=17.5),
  )
  HONDA_PASSPORT_4G = HondaBoschCANFDPlatformConfig(
    [HondaCarDocs("Honda Passport 2026", "All")],
    CarSpecs(mass=4620 * CV.LB_TO_KG, wheelbase=2.89, centerToFrontRatio=0.442, steerRatio=18.5),
  )
  ACURA_MDX_4G = HondaBoschPlatformConfig(
    [HondaCarDocs("Acura MDX 2022-24", "All", min_steer_speed=70. * CV.KPH_TO_MS)],
    CarSpecs(mass=4788 * CV.LB_TO_KG, wheelbase=2.89, steerRatio=15.8, centerToFrontRatio=0.428),  # as spec
    {Bus.pt: 'honda_common_canfd_generated'}, # not CANFD car but shares same dbc
    flags=HondaFlags.BOSCH_ALT_RADAR | HondaFlags.BOSCH_TJA_CONTROL,
  )
  # mid-model refresh
  ACURA_MDX_4G_MMR = HondaBoschCANFDPlatformConfig(
    [HondaCarDocs("Acura MDX 2025-26", "All except Type S")],
    CarSpecs(mass=4544 * CV.LB_TO_KG, wheelbase=2.89, centerToFrontRatio=0.428, steerRatio=16.7),
  )
  HONDA_ODYSSEY_5G_MMR = HondaBoschPlatformConfig(
    [HondaCarDocs("Honda Odyssey 2021-26", "All", min_steer_speed=70. * CV.KPH_TO_MS)],
    CarSpecs(mass=4590 * CV.LB_TO_KG, wheelbase=3.00, steerRatio=19.4, centerToFrontRatio=0.41),
    {Bus.pt: 'acura_rdx_2020_can_generated'},
    flags=HondaFlags.BOSCH_ALT_BRAKE | HondaFlags.BOSCH_ALT_RADAR,
  )
  ACURA_TLX_2G = HondaBoschPlatformConfig(
    [HondaCarDocs("Acura TLX 2021-23", "All")],
    CarSpecs(mass=3982 * CV.LB_TO_KG, wheelbase=2.87, steerRatio=14.0, centerToFrontRatio=0.43),
    {Bus.pt: 'honda_civic_hatchback_ex_2017_can_generated'},
    flags=HondaFlags.BOSCH_ALT_RADAR,
  )
  # mid-model refresh
  ACURA_TLX_2G_MMR = HondaBoschCANFDPlatformConfig(
    [HondaCarDocs("Acura TLX 2024-25", "All")],
    CarSpecs(mass=3990 * CV.LB_TO_KG, wheelbase=2.87, centerToFrontRatio=0.43, steerRatio=13.7),
  )
  HONDA_FIT_4G = HondaBoschPlatformConfig(
    [
      HondaCarDocs("Honda Fit (Taiwan) 2021", "All"),
      # TODO: add 2022-2023 fingerprints
      HondaCarDocs("Honda Fit (Taiwan) 2024-25", "All"),
    ],
    CarSpecs(mass=1229, wheelbase=2.53, steerRatio=19.7, centerToFrontRatio=0.39, minSteerSpeed=23. * CV.KPH_TO_MS),
    {Bus.pt: 'honda_bosch_radarless_generated'},
    flags=HondaFlags.BOSCH_RADARLESS | HondaFlags.LKAS_MINSPEED_CUTOFF
  )
  ACURA_INTEGRA = HondaBoschPlatformConfig(
    [
      HondaCarDocs("Acura Integra 2023-26", "All"),
      HondaCarDocs("Honda Prelude 2026", "All"),
    ],
    CarSpecs(mass=3338.8 * CV.LB_TO_KG, wheelbase=2.5, centerToFrontRatio=0.5, steerRatio=16.71, tireStiffnessFactor=0.82),
    {Bus.pt: 'honda_bosch_radarless_generated'},
    flags=HondaFlags.BOSCH_RADARLESS
  )
  ACURA_ADX = HondaBoschPlatformConfig(
    [HondaCarDocs("Acura ADX 2025-26", "All")],
    CarSpecs(mass=3578 * CV.LB_TO_KG, wheelbase=2.65, steerRatio=16.6, centerToFrontRatio=0.43),
    {Bus.pt: 'honda_bosch_radarless_generated'},
    flags=HondaFlags.BOSCH_RADARLESS
  )

  # Nidec Cars
  ACURA_ILX = HondaNidecPlatformConfig(
    [
      HondaCarDocs("Acura ILX 2016-18", "Technology Plus Package or AcuraWatch Plus", min_steer_speed=25. * CV.MPH_TO_MS),
      HondaCarDocs("Acura ILX 2019-22", "All", min_steer_speed=25. * CV.MPH_TO_MS),
    ],
    CarSpecs(mass=3095 * CV.LB_TO_KG, wheelbase=2.67, steerRatio=18.61, centerToFrontRatio=0.37, tireStiffnessFactor=0.72),  # 15.3 is spec end-to-end
    radar_dbc_dict('acura_ilx_2016_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES | HondaFlags.HAS_ALL_DOOR_STATES,
  )
  HONDA_CRV = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda CR-V 2015-16", "Touring Trim", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=3572 * CV.LB_TO_KG, wheelbase=2.62, steerRatio=16.89, centerToFrontRatio=0.41, tireStiffnessFactor=0.444),  # as spec
    radar_dbc_dict('honda_crv_touring_2016_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES | HondaFlags.HAS_ALL_DOOR_STATES,
  )
  HONDA_CRV_EU = HondaNidecPlatformConfig(
    [],  # Euro version of CRV Touring, don't show in docs
    HONDA_CRV.specs,
    radar_dbc_dict('honda_crv_touring_2016_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES | HondaFlags.HAS_ALL_DOOR_STATES,
  )
  HONDA_CRV_SA = HondaNidecPlatformConfig(
    [],  # South Africa version of CRV Touring, don't show in docs
    HONDA_CRV.specs,
    radar_dbc_dict('acura_rdx_2018_can_generated'), # different gearbox message from USA CRV
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES | HondaFlags.HAS_ALL_DOOR_STATES,
  )
  HONDA_FIT = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Fit 2018-20", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=2644 * CV.LB_TO_KG, wheelbase=2.53, steerRatio=13.06, centerToFrontRatio=0.39, tireStiffnessFactor=0.75),
    radar_dbc_dict('acura_ilx_2016_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES
  )
  HONDA_FREED = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Freed 2020", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=3086. * CV.LB_TO_KG, wheelbase=2.74, steerRatio=13.06, centerToFrontRatio=0.39, tireStiffnessFactor=0.75),  # mostly copied from FIT
    radar_dbc_dict('acura_ilx_2016_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_HRV = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda HR-V 2019-22", min_steer_speed=12. * CV.MPH_TO_MS)],
    HONDA_HRV_3G.specs,
    radar_dbc_dict('acura_ilx_2016_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  HONDA_ODYSSEY = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Odyssey 2018-20")],
    CarSpecs(mass=1900, wheelbase=3.0, steerRatio=14.35, centerToFrontRatio=0.41, tireStiffnessFactor=0.82),
    radar_dbc_dict('honda_odyssey_exl_2018_generated'),
    flags=HondaFlags.NIDEC_ALT_PCM_ACCEL | HondaFlags.HAS_ALL_DOOR_STATES,
  )
  HONDA_ODYSSEY_TWN = HondaNidecPlatformConfig(
    [
      HondaCarDocs("Honda Odyssey (Taiwan) 2018-19"),
      HondaCarDocs("Honda Odyssey (Singapore) 2021")
    ],
    CarSpecs(mass=1865, wheelbase=2.9, steerRatio=14.35, centerToFrontRatio=0.44, tireStiffnessFactor=0.82),
    radar_dbc_dict('honda_odyssey_twn_2018_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  ACURA_RDX = HondaNidecPlatformConfig(
    [HondaCarDocs("Acura RDX 2016-18", "AcuraWatch Plus or Advance Package", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=3925 * CV.LB_TO_KG, wheelbase=2.68, steerRatio=15.0, centerToFrontRatio=0.38, tireStiffnessFactor=0.444),  # as spec
    radar_dbc_dict('acura_rdx_2018_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES | HondaFlags.HAS_ALL_DOOR_STATES,
  )
  HONDA_PILOT = HondaNidecPlatformConfig(
    [
      HondaCarDocs("Honda Pilot 2016-22", min_steer_speed=12. * CV.MPH_TO_MS),
      HondaCarDocs("Honda Passport 2019-25", "All", min_steer_speed=12. * CV.MPH_TO_MS),
    ],
    CarSpecs(mass=4278 * CV.LB_TO_KG, wheelbase=2.86, centerToFrontRatio=0.428, steerRatio=16.0, tireStiffnessFactor=0.444),  # as spec
    radar_dbc_dict('acura_ilx_2016_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES | HondaFlags.HAS_ALL_DOOR_STATES,
  )
  HONDA_RIDGELINE = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Ridgeline 2017-26", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=4515 * CV.LB_TO_KG, wheelbase=3.18, centerToFrontRatio=0.41, steerRatio=15.59, tireStiffnessFactor=0.444),  # as spec
    radar_dbc_dict('acura_ilx_2016_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES | HondaFlags.HAS_ALL_DOOR_STATES,
  )
  HONDA_CIVIC = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Civic 2016-18", min_steer_speed=12. * CV.MPH_TO_MS, video="https://youtu.be/-IkImTe1NYE")],
    CarSpecs(mass=1326, wheelbase=2.70, centerToFrontRatio=0.4, steerRatio=15.38),  # 10.93 is end-to-end spec
    radar_dbc_dict('honda_civic_touring_2016_can_generated'),
    flags=HondaFlags.HAS_ALL_DOOR_STATES
  )

  # port extensions
  HONDA_ACCORD_9G = HondaNidecPlatformConfig(
    [
      HondaCarDocs("Honda Accord 2016-17"),
      HondaCarDocs("Honda Accord Hybrid 2017", "All"),
    ],
    CarSpecs(mass=3343 * CV.LB_TO_KG, wheelbase=2.78, steerRatio=17.5, centerToFrontRatio=0.37),  # as spec
    radar_dbc_dict('honda_accord_2017_can_ext_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES | HondaFlags.HAS_ALL_DOOR_STATES,
  )
  HONDA_CLARITY = HondaNidecPlatformConfig(
    [HondaCarDocs("Honda Clarity 2018-21", min_steer_speed=12. * CV.MPH_TO_MS)],
    CarSpecs(mass=1834, wheelbase=2.75, centerToFrontRatio=0.4, steerRatio=16.5),
    radar_dbc_dict('honda_clarity_hybrid_2018_can_generated'),
    flags=HondaFlags.HAS_ALL_DOOR_STATES,
  )
  ACURA_MDX_3G = HondaNidecPlatformConfig(
    [
      HondaCarDocs("Acura MDX 2014-16", "Advance Package"),
      HondaCarDocs("Acura MDX 2017-19", "All"),
      HondaCarDocs("Acura MDX Hybrid 2017-19", "All"),
    ],
    CarSpecs(mass=4215 * CV.LB_TO_KG, wheelbase=2.82, steerRatio=16.8, centerToFrontRatio=0.428),  # as spec, learned steerRatio
    radar_dbc_dict('acura_mdx_2017_can_ext_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  ACURA_MDX_3G_MMR = HondaNidecPlatformConfig(
    [
      HondaCarDocs("Acura MDX 2020", "All"),
      HondaCarDocs("Acura MDX Hybrid 2020", "All"),
    ],
    CarSpecs(mass=4215 * CV.LB_TO_KG, wheelbase=2.82, steerRatio=16.8, centerToFrontRatio=0.428),  # as spec, learned steerRatio
    radar_dbc_dict('acura_ilx_2016_can_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES,
  )
  ACURA_TLX_1G = HondaNidecPlatformConfig(
    [
      HondaCarDocs("Acura TLX 2015-17", "Advance Package"),
      HondaCarDocs("Acura TLX 2018-20", "All"),
    ],
    CarSpecs(mass=3680 * CV.LB_TO_KG, wheelbase=2.78, steerRatio=17.0, centerToFrontRatio=0.40, tireStiffnessFactor=0.18),
    radar_dbc_dict('acura_mdx_2017_can_ext_generated'),
    flags=HondaFlags.NIDEC_ALT_SCM_MESSAGES | HondaFlags.HAS_ALL_DOOR_STATES,
  )


HONDA_NIDEC_ALT_PCM_ACCEL = CAR.with_flags(HondaFlags.NIDEC_ALT_PCM_ACCEL)
HONDA_NIDEC_ALT_SCM_MESSAGES = CAR.with_flags(HondaFlags.NIDEC_ALT_SCM_MESSAGES)
HONDA_BOSCH = CAR.with_flags(HondaFlags.BOSCH)
HONDA_BOSCH_RADARLESS = CAR.with_flags(HondaFlags.BOSCH_RADARLESS)
HONDA_BOSCH_CANFD = CAR.with_flags(HondaFlags.BOSCH_CANFD)
HONDA_BOSCH_ALT_RADAR = CAR.with_flags(HondaFlags.BOSCH_ALT_RADAR)
HONDA_BOSCH_TJA_CONTROL = CAR.with_flags(HondaFlags.BOSCH_TJA_CONTROL)
HONDA_LKAS_MINSPEED_CUTOFF = CAR.with_flags(HondaFlags.LKAS_MINSPEED_CUTOFF)


DBC = CAR.create_dbc_map()


STEER_THRESHOLD = {
  # default is 1200, overrides go here
  CAR.ACURA_RDX: 400,
  CAR.HONDA_CRV_EU: 400,
  CAR.HONDA_ACCORD_11G: 600,
  CAR.HONDA_PILOT_4G: 600,
  CAR.HONDA_PASSPORT_4G: 600,
  CAR.ACURA_MDX_4G_MMR: 600,
  CAR.HONDA_CRV_6G: 600,
  CAR.HONDA_CITY_7G: 600,
  CAR.HONDA_PASSPORT_4G: 600,
  CAR.HONDA_NBOX_2G: 600,
  CAR.HONDA_ODYSSEY_5G_MMR: 600,
  # port extensions
  CAR.HONDA_ACCORD_9G: 30,
  CAR.ACURA_MDX_3G: 30,
  CAR.ACURA_MDX_3G_MMR: 30,
  CAR.ACURA_TLX_1G: 30,
}


HONDA_ALT_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xF112)
HONDA_ALT_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(0xF112)


FW_QUERY_CONFIG = FwQueryConfig(
  requests=[
    # Currently used to fingerprint
    Request(
      [StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.UDS_VERSION_RESPONSE],
      bus=1,
    ),

    # Data collection requests:
    # Log manufacturer-specific identifier for current ECUs
    Request(
      [HONDA_ALT_VERSION_REQUEST],
      [HONDA_ALT_VERSION_RESPONSE],
      bus=1,
      logging=True,
    ),
    # Nidec PT bus
    Request(
      [StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.UDS_VERSION_RESPONSE],
      bus=0,
    ),
    # Bosch PT bus
    Request(
      [StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.UDS_VERSION_RESPONSE],
      bus=1,
      obd_multiplexing=False,
    ),
  ],
  # We lose these ECUs without the comma power on these cars.
  # Note that we still attempt to match with them when they are present
  # This is or'd with (ALL_ECUS - ESSENTIAL_ECUS) from fw_versions.py
  non_essential_ecus={
    Ecu.eps: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_E, CAR.HONDA_E_ADVANCE, CAR.ACURA_MDX_4G, CAR.HONDA_CRV_SA, CAR.ACURA_MDX_3G,
              CAR.HONDA_ACCORD_9G, *HONDA_BOSCH_ALT_RADAR, *HONDA_BOSCH_RADARLESS, *HONDA_BOSCH_CANFD],
    Ecu.vsa: [CAR.ACURA_RDX_3G, CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_5G, CAR.HONDA_CRV_HYBRID, CAR.HONDA_E,
              CAR.HONDA_E_ADVANCE, CAR.HONDA_INSIGHT, CAR.HONDA_NBOX_2G, CAR.ACURA_MDX_4G, CAR.HONDA_ACCORD_9G,
              *HONDA_BOSCH_ALT_RADAR, *HONDA_BOSCH_RADARLESS, *HONDA_BOSCH_CANFD],
  },
  extra_ecus=[
    (Ecu.combinationMeter, 0x18da60f1, None),
    (Ecu.programmedFuelInjection, 0x18da10f1, None),
    # The only other ECU on PT bus accessible by camera on radarless Civic
    # This is likely a manufacturer-specific sub-address implementation: the camera responds to this and 0x18dab0f1
    # Unclear what the part number refers to: 8S103 is 'Camera Set Mono', while 36160 is 'Camera Monocular - Honda'
    # TODO: add query back, camera does not support querying both in parallel and 0x18dab0f1 often fails to respond
    # (Ecu.unknown, 0x18DAB3F1, None),
  ],
)
