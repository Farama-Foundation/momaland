from pettingzoo.test import parallel_api_test

from momadm_benchmarks.envs.beach_domain.beach_domain import MOBeach


def test_parallel_env():
    parallel_api_test(MOBeach())
