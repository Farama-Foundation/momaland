"""Environment registry.

Used for:
- testing
- rendering GIF images
"""

from momaland.envs.beach_domain import mobeach_v0
from momaland.envs.crazyrl.catch import mocatch_v0
from momaland.envs.crazyrl.escort import moescort_v0
from momaland.envs.crazyrl.surround import mosurround_v0
from momaland.envs.item_gathering import moitemgathering_v0
from momaland.envs.multiwalker import momultiwalker_v0


all_environments = {
    "mobeach_v0": mobeach_v0,
    "momultiwalker_v0": momultiwalker_v0,
    "mocatch_v0": mocatch_v0,
    "mosurround_v0": mosurround_v0,
    "moescort_v0": moescort_v0,
    "moitemgathering_v0": moitemgathering_v0,
}
