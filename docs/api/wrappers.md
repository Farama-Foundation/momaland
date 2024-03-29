---
title: "Wrappers and Utils"
---

# Wrappers and Utils

A wrapper is an environment transformation that takes in an environment as input, and outputs a new environment that is similar to the input environment, but with some transformation or validation applied.

For conversion between `AEC` and `Parallel` APIs, the native MOMAland wrappers must be used. On top of conversion wrappers, there are also a few utility wrappers.

Wrappers for the `AEC` and `Parallel` wrappers are split into their own modules and can be accessed like `momaland.utils.parallel_wrappers.LinearizeReward`.

## Conversion

### `AEC to Parallel`

```{eval-rst}
.. autoclass:: momaland.utils.conversions.mo_aec_to_parallel_wrapper
```

### `Parallel to AEC`

```{eval-rst}
.. autoclass:: momaland.utils.conversions.mo_parallel_to_aec_wrapper
```

## `AEC`

```{eval-rst}
.. autoclass:: momaland.utils.aec_wrappers.LinearizeReward
```

```{eval-rst}
.. autoclass:: momaland.utils.aec_wrappers.NormalizeReward
```

## `Parallel`

```{eval-rst}
.. autoclass:: momaland.utils.parallel_wrappers.LinearizeReward
```

```{eval-rst}
.. autoclass:: momaland.utils.parallel_wrappers.NormalizeReward
```

```{eval-rst}
.. autoclass:: momaland.utils.parallel_wrappers.RecordEpisodeStatistics
```
