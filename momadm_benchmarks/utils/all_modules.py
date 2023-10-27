"""Environment registry.

Used for:
- testing
- rendering GIF images
"""

from momadm_benchmarks.envs.beach_domain import mobeach_v0
from momadm_benchmarks.envs.crazyrl.catch import mocatch_v0
from momadm_benchmarks.envs.crazyrl.escort import moescort_v0
from momadm_benchmarks.envs.crazyrl.surround import mosurround_v0
from momadm_benchmarks.envs.multiwalker import momultiwalker_v0


all_environments = {
    "mobeach_v0": mobeach_v0,
    "momultiwalker_v0": momultiwalker_v0,
    "mocatch_v0": mocatch_v0,
    "mosurround_v0": mosurround_v0,
    "moescort_v0": moescort_v0,
}
