#!/usr/bin/env python3
"""Regenerate `nidec_brake_revert_golden.npz`: the pre-change NIDEC_ALT brake trace.

`test_honda_nidec_brake.TestExactRevert` asserts that the three 2026-07-25 brake-channel
knobs (`NIDEC_GRADE_CREDIT_CLAMP`, `NIDEC_FRIC_MIN_AMP`, `NIDEC_BRAKE_MIN_DWELL`) at
their off values reproduce the controller as it stood BEFORE that change, frame for
frame.  That reference cannot be computed from the working tree, so it is captured once
and committed.

How the reference is built: the pre-change `carcontroller.py` text is read out of git and
exec'd as an anonymous module.  It still imports the CURRENT `values.py` -- deliberately.
The pre-change controller reads none of the new parameters, and every parameter it does
read is unchanged, so this isolates exactly the carcontroller edit.

  ./gen_nidec_brake_golden.py <git-ref>     # default: the commit before the change

Only re-run this when the reference itself must move (i.e. some other change to the
NIDEC brake path is deliberately accepted), and say so in the commit message.
"""
import argparse
import os
import subprocess
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
SRC = 'opendbc/car/honda/carcontroller.py'
OUT = os.path.join(HERE, 'nidec_brake_revert_golden.npz')


def load_reference_carcontroller(ref):
  src = subprocess.check_output(['git', '-C', REPO, 'show', f'{ref}:{SRC}'], text=True)
  for token in ('NIDEC_GRADE_CREDIT_CLAMP', 'NIDEC_FRIC_MIN_AMP', 'NIDEC_BRAKE_MIN_DWELL',
                'NIDEC_BRAKE_HOLD_RELEASE_A', 'NIDEC_BRAKE_HOLD_RELEASE_T',
                'brake_hold_frames', 'brake_hold_ramp', 'brake_hold_pos_frames', 'hold_floor'):
    if token in src:
      raise SystemExit(f"{ref}:{SRC} already contains {token} -- that is not a pre-change ref")
  mod = types.ModuleType('honda_carcontroller_reference')
  mod.__file__ = f'{SRC}@{ref}'
  exec(compile(src, mod.__file__, 'exec'), mod.__dict__)

  # same stub the test rig uses: HondaParamWriter starts a daemon thread and holds a
  # Params() handle per controller, and only persists learned factors every 6000 frames
  class _NoParamWriter:
    def __init__(self, *a, **k):
      pass

    def put_many(self, values):
      pass

  mod.HondaParamWriter = _NoParamWriter
  return mod.CarController


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('ref', nargs='?', default='HEAD', help='git ref holding the PRE-change source')
  args = ap.parse_args()

  sys.path.insert(0, REPO)
  from opendbc.car.honda.tests import test_honda_nidec_brake as T

  RefCarController = load_reference_carcontroller(args.ref)
  rig = T.Rig()                     # parameters are irrelevant: the reference ignores them
  rig.cc = RefCarController({T.Bus.pt: T.CAR.HONDA_ODYSSEY.config.dbc_dict[T.Bus.pt]},
                            rig.cc.CP, rig.cc.CP_SP)
  trace = T.run_sequence(rig)
  np.savez_compressed(OUT, trace=trace, ref=np.array([args.ref]))
  print(f'wrote {OUT}  shape {trace.shape}  from {args.ref}:{SRC}')
  print(f'  brake counts: nonzero on {int((trace[:, 0] > 0).sum())} frames, max {trace[:, 0].max():.0f}')
  print(f'  pcm_off     : min {trace[:, 1].min():+.3f}  max {trace[:, 1].max():+.3f}')


if __name__ == '__main__':
  main()
