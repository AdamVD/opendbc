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
  # Demand-ramped engine-brake credit -- REVERTED OFF same evening (2026-07-11 pm, Adam's
  # pushback CONFIRMED by lift_plant_check.py, 28k friction-free grade-corrected frames):
  # flat-grade engine contribution at a FULL commanded lift (po -1.1..-1.6) is only p50
  # -0.08..-0.11 (p10 -0.19); shallow lift (po -0.2..-0.6) is +0.08 (residual throttle);
  # downhill -0.14..-0.17. The briefly-shipped EB_V=[0.20,0.30] was based on the 7/11
  # decel-hold x2 read, which decomposes into friction-active windows + a window-min metric
  # artifact -- NOT flat-grade lift authority. A 0.2-0.3 credit re-creates the documented
  # pre-6/10 downhill deficit (+0.27, "P1"): descents lose friction with no compensating
  # engine brake (stock gets its descent engine-braking from TCU DOWNSHIFTS our constant
  # vEgo re-anchoring never provokes -- that is the separate descent-mode spec,
  # FINDINGS stock grade 2026-07-06). Machinery kept for that work; measured honest values
  # if ever re-armed: EB_V ~[0.08, 0.10] flat. EB_DYN=False = flat 0.05 credit (6/10 ship).
  NIDEC_MODEL_EB_DYN = False
  NIDEC_MODEL_EB_BP = [10., 30.]    # m/s
  NIDEC_MODEL_EB_V = [0.08, 0.10]   # m/s^2 credit at full lift (MEASURED flat-grade, lift_plant_check)
  NIDEC_MODEL_EB_FULL_AT = 0.40     # m/s^2 demand at which the lift is ~floored (|a_des| ramp end)

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
  NIDEC_DITHER_AMP = 0.0        # servo-ID dither amplitude (m/s pcm_off). 0.0 = INERT (default).
                                # Experiment value: 0.3 (the validated aEgo~0 null band edge). Only
                                # applies in calm no-lead cruise; see carcontroller gating. Purpose:
                                # persistent excitation for servo tau + small-signal s[gear] ID
                                # (plant-simulator campaign 2026-07-24). Flip for ONE supervised
                                # 20-30 min highway drive, then back to 0.0.
  NIDEC_DITHER_DWELL_S = 0.8    # dither PRBS chip duration (s). 0.8 s spans the servo tau range
                                # (0.2-2 s) with energy on both sides.
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

  # Servo/saturation-aware FF (2026-06-17, replay-validated -> FINDINGS_servo_ff_replay_2026-06-17.md).
  # The inverse-K map pcm_off=a_des/K(v) inverts a LINEAR plant, but the Honda PCM is a saturating
  # SPEED SERVO and pcm_off IS the commanded speed error: the servo floors throttle at the accel
  # ceiling whenever the error exceeds ~PO_EDGE, so a_des/K (K~0.13) inflates the offset to 5-8 in the
  # feedback-INERT saturated zone (cap-counterfactual: aego identical for pcm_off 4..8). The car then
  # keeps pulling ~1-2s after the planner eases, because the offset must ooze back below the band edge
  # before the servo responds (the "two shelves" follow-pull complaint, 2026-06-16). Fix: while EASING
  # invert the servo's REAL proportional gain Ks=CEIL(v)/PO_EDGE so the command lands in the responsive
  # band, and park the offset at the band edge while saturated -> the servo eases the instant the
  # planner asks below the ceiling. Onset is UNCHANGED (full 1/K oomph while the request is building,
  # gated by an a_des trend latch, so rising hysteresis still gets punched through) and decel (a_des<0)
  # is byte-for-byte the DECEL_SOFT path, so the anti-slam grading and the relay zone are untouched.
  # Replay (real MPC in loop): railed accel shelf shortens 0.87s open-loop / ~1.5s closed-loop on ep38,
  # lead gap opens, no in-sim hunting. CAVEAT: the throttle-slam relay was killed by DECEL_SOFT before
  # these logs existed, so replay CANNOT certify it stays gone -- that is the ON-CAR A/B gate. Set
  # NIDEC_MODEL_SERVO_AWARE=False to revert to the exact inverse-K baseline (the 2026-06-16 map).
  NIDEC_MODEL_SERVO_AWARE = True    # False = exact a_des/K(v) baseline; True = servo-aware ease
  NIDEC_MODEL_PO_EDGE = 1.8         # m/s; servo proportional-band edge (saturation knee). aego rails
                                    # at the ceiling for pcm_off>~PO_EDGE and modulates below it
                                    # (falling-release ~1.5, rising-onset ~3.5; hysteresis straddles).
  NIDEC_MODEL_PO_HOLD_MARGIN = 0.4  # while saturated, park offset at PO_EDGE+this (=2.2) -- margin
                                    # above the knee so sensor/plan noise can't trip a premature ease.
  # Accel CEILING = the Honda authority plateau aego cannot exceed vs speed (NOT the planner's
  # A_CRUISE_MAX): ~0.89 m/s^2 low-speed, ~0.55 highway (accel-delivery-ceiling 2026-06-08 + the
  # 2026-06-17 plant-ID). Tune on-car: raise if onset feels soft, lower toward the measured plateau
  # if the ease still lags. ceil(v)=clip(CEIL_V0 - CEIL_K*v, CEIL_MIN, CEIL_MAX).
  NIDEC_MODEL_CEIL_V0 = 0.975       # ceiling at 0 m/s (m/s^2)
  NIDEC_MODEL_CEIL_K = 0.0142       # ceiling falloff (m/s^2 per m/s)
  NIDEC_MODEL_CEIL_MIN = 0.35       # clamp
  NIDEC_MODEL_CEIL_MAX = 0.95       # clamp
  NIDEC_MODEL_SERVO_TREND_TAU = 0.4   # s; a_des trend low-pass for the build-vs-ease latch
  NIDEC_MODEL_SERVO_TREND_EPS = 0.02  # m/s^2; trend deadband (hysteresis) -> no build/ease chatter

  # DBO -- onset front-load + sustain-gated recoverability cap on the build-side pcm_off
  # (2026-06-19; FINDINGS_knee_is_downshift_2026-06-19.md). Applied ON TOP of the servo-aware FF;
  # decel/brake untouched. Two halves implementing "induce torque faster, but never to an extent
  # that puts us in an unrecoverable over-accel", grounded in the 0x130 plant-ID (the pcm_off knee
  # IS a downshift; cruise gear has ~0 gentle-accel authority -- accel ONLY comes from a kickdown):
  #
  #  ONSET FRONT-LOAD ("induce torque faster"): for a GENUINE rising accel demand (po_build &&
  #  a_des > ONSET_MIN) the baseline servo-aware FF parks pcm_off in its ease band (~1.5, BELOW the
  #  ~2.5 knee) for a gradual gap-close -> it SAGS, delivering no torque until the gap opens enough
  #  to force a fast ramp. DBO instead crosses the knee decisively to KNEE_CROSS so the (mild,
  #  wanted) downshift fires PROMPTLY -> torque arrives ~0.5s sooner (open-loop demo onset_test).
  #  torque_req has no dead-time (0x130 Test C) so the cross translates immediately to a torque
  #  request. Gated to genuine demand so micro-nudges don't fire downshifts; the slew ramps the lift
  #  (no 1-frame jump). HONEST: this commits to torque more readily = MORE (shallow) downshifts --
  #  the intended trade for faster response (Adam's "induce torque faster"); and the TCU honors the
  #  cross only ~partially. The downshift FREQUENCY effect is unquantifiable in sim -> on-car A/B.
  #
  #  RECOVERABILITY CAP ("never unrecoverable over-accel"): baseline rails pcm_off to ~5-8, but
  #  pcm_off > ~5 is INERT (aego identical 4..8; authority saturates ~3-4 -- Adam + plant-ID) and
  #  the railed offset oozes back over 1-2s = the CONFIRMED over-pull/overshoot (Jekyll Check-2).
  #  Cap at PO_CAP -> bounds the pull (gear-7 at pcm_off 3.2 ~0.35 vs railed ~0.6) and retracts in
  #  ~one 0.85s lag -> the "extremely strong overshoot" of a deep kickdown is TAMED. The cap does
  #  NOT prevent the downshift DEPTH (the 10->7 itself; depth is TCU/rate-internal, pcm_off LEVEL is
  #  flat ~2.3 across depths -- knee_depth.py); preventing the 10->7 needs a follow gear-hold,
  #  DEFERRED. SUSTAIN-GATE: cap=PO_CAP for nudges; demand persisting > SUSTAIN_T uncaps to FULL_CAP
  #  so a real set-speed bump / lead pull-away keeps full authority.
  #
  # Reviewed by 2 adversarial subagents + advisor: a -25% downshift-FREQUENCY claim was a sim
  # artifact (RETRACTED -- the gear-sim cannot measure frequency); the cap's overshoot benefit rests
  # on the confirmed mechanism, not a sim number. Sim is DIRECTIONAL; on-car A/B vs the servo-aware
  # baseline is the gate. NIDEC_MODEL_DBO=False is byte-identical to the 2026-06-17 baseline.
  NIDEC_MODEL_DBO = True   # RE-ARMED 2026-07-11 as CAP-ONLY (FRONT_LOAD=False below) alongside the
                           # planner's opening-gap chase governor (FINDINGS_follow_policy_2026-07-09.md
                           # rev 7/10): governor bounds the ask upstream (~0.35 in opening-gap follow);
                           # the cap de-inflates whatever still crosses the knee (9.6% of follow frames
                           # pre-governor, ask p98 4.47). Was DISABLED 2026-06-21 (Adam's call, never
                           # driven on the Jekyll trip). False remains byte-identical to the 2026-06-17
                           # servo-aware baseline.
  NIDEC_DBO_FRONT_LOAD = False  # onset front-load ("induce torque faster") DISABLED 2026-07-11: it forces
                              # pcm_off >= KNEE_CROSS for any modest rising ask -- exactly the knee-cross
                              # that FINDINGS_sustained_hold_overshoot_2026-07-10.md identified as the
                              # "0.3 held long enough gets 0.6" mechanism (59% of >=6s holds downshift;
                              # no-shift holds track fine), and it chases the opening gap the 7/9 follow-
                              # policy work says factory deliberately does NOT chase. Cap-only was always
                              # the solid half (6/19 advisor review); front-load stays an unproven bet.
  NIDEC_DBO_KNEE_CROSS = 2.9  # m/s; onset front-load target -- cross the soft knee (~2.5) decisively
                              # so the mild downshift fires promptly (inert while FRONT_LOAD=False).
  NIDEC_DBO_ONSET_MIN = 0.15  # m/s^2; only front-load a RISING a_des above this (genuine demand;
                              # micro-nudges below it stay on the servo-aware FF -> no downshift spam).
  NIDEC_DBO_PO_CAP = 3.2      # m/s; nudge cap (recoverability). Tune DOWN toward ~3.0 on-car if the
                              # capped pull still overshoots; UP if onset feels gutless on follow.
  NIDEC_DBO_FULL_CAP = 5.0    # m/s; sustained-demand cap (>5 is inert, only deepens the kickdown).
  NIDEC_DBO_SUSTAIN_T = 1.3   # s; pcm_off above PO_CAP for this long -> uncap to FULL_CAP.
  # Sustain-uncap a_des gate (2026-07-10 fix, revised on review 2026-07-11): the time gate alone
  # tripped on 53% of patient >=6s follow holds (hold_dbo_check.py), handing authority back exactly
  # when the cap was needed; a real set-speed bump / pull-away asks well above a patient follow
  # hold's 0.2-0.45. a_des includes the grade FF term, so sustained uphill demand still uncaps and
  # keeps climb authority. Speed-dependent (7/11): a fixed 0.5 was unreachable above ~36 m/s where
  # the planner's own A_CRUISE_MAX caps asks at 0.54..0.50 -- the threshold now tracks that ceiling
  # (a railed ask IS full demand). HYST (7/11): counter FREEZES (neither counts nor resets) while
  # a_des sits within HYST below the threshold with demand still above PO_CAP, so unfiltered-pitch
  # dither can't zero the 1.3s counter or flap an earned uncap; a_des dropping below the band or
  # pcm_off easing under PO_CAP still resets normally.
  NIDEC_DBO_UNCAP_A_BP = [25., 40.]  # m/s
  NIDEC_DBO_UNCAP_A_V = [0.5, 0.42]  # m/s^2; count-up threshold on a_des (flat 0.5 below 25 m/s)
  NIDEC_DBO_UNCAP_A_HYST = 0.1       # m/s^2; freeze band below the threshold
  # Knee guard (2026-07-11 first-drive eval, FINDINGS_gov_first_drive). PO_CAP=3.2 bounds kickdown
  # DEPTH but sits ABOVE the ~2.5 downshift knee, so a small ask (0.2-0.35, incl. every governor-
  # capped chase ask) still maps via 1/K to pcm_off 2-3, crosses the knee, and delivers x1.9-2.4
  # (54 small holds: ask p50 0.22 -> delivered 0.54; governed surges: aTarget pinned 0.35 ->
  # delivered 0.68-0.84 -> the limit cycle survived the governor). While demand is below the
  # sustain-uncap threshold, park the command at the servo band edge (PO_EDGE+HOLD_MARGIN=2.2,
  # same point the servo-aware ease already uses) -- small asks live in the proportional band and
  # CANNOT fire the downshift; demand above uncap_a keeps today's PO_CAP/FULL_CAP ladder (a_des
  # includes grade FF, so uphill asks retain authority). Same hysteresis band as the uncap gate
  # (latch holds inside it) so pitch dither can't flap the guard. 0.0 disables (exact 7/11 ship).
  NIDEC_DBO_KNEE_GUARD = 2.2  # m/s; pcm_off cap while a_des < uncap_a (0 = off)
  # Knee-guard input (2026-07-18 uphill-follow root cause, FINDINGS_uphill_follow_2026-07-18):
  # testing the guard on grade-FF-INFLATED a_des released it on every steep-grade chase -- the
  # capped 0.35 ask + 2.2*sin(theta) crossed uncap_a exactly where the downshift's consequences
  # are worst (retained-low-gear limit cycle, ~10s period, route c2 @6%). RAW=True latches the
  # guard on actuators.accel instead: a genuine climb still releases it (speed sag drives the
  # raw PID ask past uncap_a within a few s), and the PCM self-downshifts under load at the
  # 2.2 park anyway (observed) -- we just stop COMMANDING kickdown depth for grade FF alone.
  # False = exact 7/11 behavior (a_des test).
  NIDEC_DBO_KNEE_GUARD_RAW = True
  # Lead hold (2026-07-19, FINDINGS_overshoot_pull_2026-07-19): the RAW release path still fired
  # a commanded kickdown INTO the follow mark -- d3 t529 (6% climb): hill starts, guarded po parks
  # at 2.2, speed sags 3 mph, PID ask crosses uncap_a "by design", po 3.2 -> downshift lands with
  # a matched lead at THW 2.2/vRel 0 -> fresh-downshift gear pulls +0.3 for 10 s (62->70 mph
  # uphill) against a plan capped at 0 -> overshoot to THW 0.90 + cb 0.73. With a lead visible,
  # never release the knee guard or the sustain-uncap ladder past the 2.2 park: the PCM still
  # self-downshifts under sustained load at the park (observed d4 t999 and on stock -- stock never
  # commands kickdown depth either), we just don't ASK for it while there is someone to run into.
  # Open-road climbs (no lead) keep the 7/18 release exactly. False -> exact 7/18 behavior.
  NIDEC_DBO_LEAD_HOLD = True
  NIDEC_DBO_DAMP = 0.0        # m/s^2; EXPERIMENTAL, default 0 (inactive). >0 coasts a_des below it
                              # (don't command the wanted mild downshift). On-car sag<->frequency
                              # knob ONLY -- unvalidated for frequency, loosens the gap; leave at 0.

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

  # ---- Low-gear lift guard (2026-07-18, FINDINGS_uphill_follow_2026-07-18) ----
  # Gear-resolved lift plant ID (plant_gear_id_0718.py, 403k engaged frames): in top gears the
  # throttle-shut lift delivers the modeled ~+0.05..0.07 EB, but in a downshifted gear the lift
  # response is BIMODAL -- torque persists 1-3s after the command (uphill rn>=1.35 lift windows
  # show LESS decel than gravity alone, p50 -0.27..-0.40 "EB"), then snaps to fuel-cut + low-gear
  # EB (the trace-verified -1.07 delivered on a -0.5 ask, cb=0). A proportional inverse-K map
  # cannot represent that plant. Meanwhile pcm_off in [-0.3, +0.3] holds aEgo ~= 0 on ANY grade
  # (n=97k frames incl. 4-8% downshifted) -- the servo-modulation band is safe and predictable.
  # While the engine is in kickdown state (rpm latch below), a decel ask therefore:
  #   - floors pcm_off at LIFT_PO_FLOOR (stay in the modulation band; never command the snap)
  #   - zeroes the engine-brake credit (it is a lie in this state -- bimodal, unschedulable),
  #     so the friction channel (linear, measured 2.78 plant) serves what gravity cannot.
  # Skipped while descent-mode is latched/trimming (rev 3 owns that envelope byte-identically).
  # rpm=0 (signal absent) leaves the guard off -> exact prior behavior.
  NIDEC_LIFT_GUARD = True
  NIDEC_LIFT_GUARD_RPM_ON = 1900.   # rpm; latch on above (10th-gear cruise ~1450-1550 @ 75mph)
  NIDEC_LIFT_GUARD_RPM_OFF = 1650.  # rpm; release below (hysteresis vs TC-lockup dither)
  # ---- 7/19 first-drive fixes (FINDINGS_overshoot_pull_2026-07-19) ----
  # (a) LATCH LEAK: the rpm hysteresis conflated "rpm above release" with "kickdown state" --
  # RPM_OFF=1650 sits BELOW normal mid-gear cruise rpm (9th ~1660 @ 50mph, 8th ~1840 @ 67mph),
  # so one crossing of 1900 kept the guard latched through 36.4% of 7/19 engaged time, including
  # plain 9th-gear cruising where the lift plant is healthy (all 10 EASED approaches unlatched,
  # all 3 LATE_BRAKE latched -- overshoot_scan_0719.py). Latch instead on the GEAR RATIO
  # engine_rpm/xmission_speed (rpm per kph; XMISSION_SPEED is kph, factor 0.01), which clusters
  # cleanly per gear: 15.7 (10th) / 17.6 / 21.1 cruise states vs 26.9+ in the retained-kickdown
  # states that actually lift bimodally. RATIO=False -> exact 7/18 rpm-hysteresis behavior.
  NIDEC_LIFT_GUARD_RATIO = True
  NIDEC_LIFT_RATIO_ON = 24.0        # rpm/kph; latch on above (kickdown cluster 26.9; cruise max 21.1)
  NIDEC_LIFT_RATIO_OFF = 22.5       # rpm/kph; release below (hysteresis vs in-shift ratio sweep)
  NIDEC_LIFT_RATIO_XSPD_MIN = 20.0  # kph; below this the ratio is TC-slip/near-zero noise -> guard off
  # (b) PHANTOM CREDITS while clamped: the guard pins po in the hold band, i.e. the PCM
  # speed-servo HOLDS SPEED WITH THE THROTTLE OPEN -- aero (wind_ms2) and the gravity blend
  # (hill_brake_ff) are not decelerating the car, yet fric_demand still subtracted both. Light
  # decel asks (-0.05..-0.5 flat, to -0.9 on 6% up) therefore delivered ~nothing (7/19 latched
  # delivery error p50 +0.24 vs +0.04 unlatched) until the ask outgrew the credits and friction
  # served the accumulated demand at once (d4 t1038: cb 1.85 at THW 1.6; d3 t529: cb 0.73 at
  # THW 0.90 = Adam's "light steady pull ... then oversized slowdown"). While guarded-and-
  # clamped, friction serves the RAW ask: no eb/wind/hill credit (the hold band holds aEgo~0 on
  # any grade -- the servo cancels grade and aero alike). False -> exact 7/18 credit accounting.
  NIDEC_LIFT_HONEST_FRICTION = True
  # The clamp is TWO-SIDED while the raw ask is a decel: the w_passive blend inverts deep uphill
  # decel asks into positive a_des (c2 t=54s: act_a -0.37 -> wire po +3.2), which the accel paths
  # then serve as a low-gear PULL -- cap at the hold-band edge too (replay_fix_check_0718.py).
  NIDEC_LIFT_PO_FLOOR = -0.30       # m/s; deepest lift command while guarded (Q3 band edge)
  NIDEC_LIFT_PO_CEIL = 0.30         # m/s; highest command while guarded on a raw decel ask

  # ---- Descent mode: engine-brake-first hill descents (SPEC_descent_mode_2026-07-11) ----
  # Stock ACC engine-brakes grade descents via the PCM speed servo + TCU downshift and ~never
  # friction-brakes for grade (FINDINGS_stock_grade_behavior_2026-07-06: 43 windows, holds -5.6%
  # @110kph friction-free; downshifts at +0.8..+6 kph over set). Our per-frame re-anchoring
  # pcm_speed=vEgo+pcm_off never presents the growing overspeed error that fires it. While
  # latched (descending, at/over set, demand mild BECAUSE the planner's DESCENT_* floor
  # tolerates the band -- longitudinal_planner.py), anchor pcm_off=clip((set-BIAS)-vEgo,
  # PCM_OFF_MIN, 0) so PCM_SPEED holds constant while vEgo grows (stock-identical growing
  # error, pre-loaded 2 kph -> the downshift fires ~2 kph earlier = tighter than stock) and
  # zero the friction cover in-band. Friction past the band, on lead/curve demand (the a_des
  # gate releases same-frame when the planner floor lets a real ask through), or on pedal is
  # byte-identical to baseline. Trade accepted (Adam 7/11): more/earlier audible downshifts
  # than stock on gentle grades; BIAS is the single quiet-it-down knob.
  NIDEC_DESCENT = True              # False = exact prior behavior (A/B)
  NIDEC_DESCENT_BIAS = 0.56         # m/s (2 kph) anchor pre-load -- the "how tight" knob. Applied
                                    # PROPORTIONALLY to overspeed (bias_eff = min(BIAS, max(0, v_err)),
                                    # review 7/12): a constant pre-load put the anchored equilibrium
                                    # BELOW set on grades the servo can hold, sagging ~2 kph under and
                                    # flapping at the BAND_LOW exit; proportional pre-load keeps the
                                    # equilibrium AT set and ramps the downshift-trigger error only as
                                    # real overspeed develops.
  NIDEC_DESCENT_PITCH_ON = -0.012   # rad (~-1.2% grade): latch-enter threshold (LP-filtered pitch)
  NIDEC_DESCENT_PITCH_OFF = -0.008  # rad (~-0.8%): latch-exit (hysteresis; pitch dither can't flap)
  NIDEC_DESCENT_PITCH_TAU = 1.0     # s LP on pitch for the latch ONLY (FF paths keep raw pitch)
  NIDEC_DESCENT_V_MIN = 12.0        # m/s: above the 21.5 mph PCM cancel floor; corpus-validated regime
  NIDEC_DESCENT_BAND_TOP = 0.83     # m/s (+3 kph over set): band exit -> friction trims (stock rode +6)
  NIDEC_DESCENT_BAND_REARM = 0.42   # m/s (+1.5 kph): re-enter only below this (bounds the trim cycle)
  NIDEC_DESCENT_BAND_LOW = -0.5     # m/s below set: low-side exit (grade eased); the anchor may gas
                                    # back to set (pcm_off up to +|BAND_LOW|) so drag-dominant gentle
                                    # grades hold set instead of sagging into this exit (review 7/12)
  NIDEC_DESCENT_ADES_MIN = -0.35    # m/s^2 on actuators.accel (NOT a_des_b -- the grade-FF blend made
                                    # the gate self-release on steep grades, review 7/12): release when
                                    # the PID output drops past this (planner released a real demand)
  NIDEC_DESCENT_ADES_REARM = -0.25  # m/s^2: actuators.accel re-arm hysteresis
  NIDEC_DESCENT_LEAD_ADES = -0.10   # m/s^2: with a visible lead, ANY sustained decel intent past this
                                    # releases -- mild lead asks (-0.15..-0.30) must not be tolerated
                                    # (review 7/12: the -0.35 gate alone suppressed decel toward a
                                    # closing lead for seconds)
  NIDEC_DESCENT_FRIC_ENTRY = 0.05   # m/s^2: latch ENTRY requires current friction demand below this
                                    # (seamless engage -- no one-frame brake dump on re-arm)
  NIDEC_DESCENT_REARM_T = 5.0       # s: bte (band-top-exit) ALSO clears after this cooldown, not only
                                    # below BAND_REARM (rev 3, first drive 7/12: steep-grade friction
                                    # equilibrium sits at +0.5..+1.0 -- ABOVE the +0.42 re-arm line --
                                    # so one band-top excursion, e.g. a set-speed tap at a crest, locked
                                    # the latch out for the REST of the hill: 17.8 s of friction on the
                                    # a8 descent. Cooldown bounds the trim sawtooth by TIME instead of
                                    # by unreachable geometry; instant re-arm below BAND_REARM kept)
  NIDEC_DESCENT_GAS_ZERO = False    # A/B, default OFF -- premise FALSIFIED before first flip: the 7/6
                                    # stock corpus shows the camera keeps PCM_GAS PINNED AT 198 during
                                    # its own engine-brake downshifts; the trigger is the PCM servo's
                                    # growing overspeed error, NOT the gas request. Zeroing would
                                    # diverge from stock. Kept as a last-resort experiment only. Note
                                    # stock does NOT downshift on gentle (-1.5..-3%) descents (closed
                                    # throttle holds +1..2.5 kph), only at -3.9..-5.6% -- exactly the
                                    # grades where the pre-rev-3 bte lockout kept our anchor out, so
                                    # sustained engagement (REARM_T) is the real downshift lever.

  BOSCH_ACCEL_MIN = -3.5  # m/s^2
  BOSCH_ACCEL_MAX = 2.0  # m/s^2

  BOSCH_GAS_LOOKUP_BP = [0.0, 2.0]  # 2m/s^2
  BOSCH_GAS_LOOKUP_V = [0, 1600]

  STEER_STEP = 1  # 100 Hz
  STEER_DELTA_UP = 3  # min/max in 0.33s for all Honda
  STEER_DELTA_DOWN = 3
  STEER_GLOBAL_MIN_SPEED = 3 * CV.MPH_TO_MS

  def __init__(self, CP):
    # Descent-mode tuning trap (review 7/12): the anchor's max presentable error is
    # BIAS + BAND_TOP; past -PCM_OFF_MIN it silently clips and the growing-error signal
    # the whole design depends on flattens exactly where the downshift should fire.
    assert self.NIDEC_DESCENT_BIAS + self.NIDEC_DESCENT_BAND_TOP <= -self.NIDEC_MODEL_PCM_OFF_MIN

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
