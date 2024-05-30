"""Environment registry.

Used for:
- testing
- rendering GIF images
"""

from momaland.envs.beach import mobeach_v0
from momaland.envs.breakthrough import mobreakthrough_v0
from momaland.envs.connect4 import moconnect4_v0
from momaland.envs.crazyrl.catch import catch_v0
from momaland.envs.crazyrl.escort import escort_v0
from momaland.envs.crazyrl.surround import surround_v0
from momaland.envs.gem_mining import mogem_mining_v0
from momaland.envs.ingenious import moingenious_v0
from momaland.envs.item_gathering import moitem_gathering_v0
from momaland.envs.multiwalker_stability import momultiwalker_stability_v0
from momaland.envs.pistonball import mopistonball_v0
from momaland.envs.route_choice import moroute_choice_v0
from momaland.envs.samegame import mosame_game_v0


all_environments = {
    "mobeach_v0": mobeach_v0,
    "momultiwalker_stability_v0": momultiwalker_stability_v0,
    "catch_v0": catch_v0,
    "surround_v0": surround_v0,
    "escort_v0": escort_v0,
    "moitem_gathering_v0": moitem_gathering_v0,
    "moingenious_v0": moingenious_v0,
    "mopistonball_v0": mopistonball_v0,
    "moroute_choice_v0": moroute_choice_v0,
    "moconnect4_v0": moconnect4_v0,
    "mobreakthrough_v0": mobreakthrough_v0,
    "mosame_game_v0": mosame_game_v0,
    "mogem_mining_v0": mogem_mining_v0,
}
