"""Environment registry.

Used for:
- testing
- rendering GIF images
"""

from momaland.envs.beach_domain import mobeach_v0
from momaland.envs.breakthrough import mobreakthrough_v0
from momaland.envs.congestion_game import mocongestion_v0
from momaland.envs.connect4 import moconnect4_v0
from momaland.envs.crazyrl.catch import catch_v0
from momaland.envs.crazyrl.escort import escort_v0
from momaland.envs.crazyrl.surround import surround_v0
from momaland.envs.gem_mining import mogemmining_v0
from momaland.envs.item_gathering import moitemgathering_v0
from momaland.envs.multiwalker import momultiwalker_v0
from momaland.envs.pistonball import mopistonball_v0
from momaland.envs.samegame import mosamegame_v0


all_environments = {
    "mobeach_v0": mobeach_v0,
    "momultiwalker_v0": momultiwalker_v0,
    "catch_v0": catch_v0,
    "surround_v0": surround_v0,
    "escort_v0": escort_v0,
    "moitemgathering_v0": moitemgathering_v0,
    "mopistonball_v0": mopistonball_v0,
    "mocongestion_v0": mocongestion_v0,
    "moconnect4_v0": moconnect4_v0,
    "mobreakthrough_v0": mobreakthrough_v0,
    "mosamegame_v0": mosamegame_v0,
    "mogemmining_v0": mogemmining_v0
}
