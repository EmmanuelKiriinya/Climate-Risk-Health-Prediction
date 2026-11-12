"""v3 features: age-band x era interactions on the CORRECT slicing.

Rounds 1-5 tested `is_under5 (age<5) x year` and found it hurt. That merges two
bands with opposite behaviour:
    age==0 : target rate 0.966 (<=2017) -> 0.563 (>=2018)   [-0.40]
    age 1-4: target rate 0.927 (<=2017) -> 0.875 (>=2018)   [-0.05]
Averaging a -0.40 effect with a -0.05 one erases the signal. Same for season:
the monthly swing (0.35-0.50) lives only in the age>=5 group, but was tested as
`u5 x season`. This module slices on age==0 / 1-4 / 5+ instead.
"""
import numpy as np
import pandas as pd


def add_v3(X: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    age = raw["age"].values
    yr = raw["deathdate"].dt.year.values
    month = raw["deathdate"].dt.month.values

    a0 = (age == 0).astype(float)
    a14 = ((age >= 1) & (age < 5)).astype(float)
    a5p = (age >= 5).astype(float)
    post = (yr >= 2018).astype(float)

    out["is_age0"] = a0
    out["is_age1_4"] = a14
    out["post2018"] = post

    # the interaction that matters: infants behave completely differently by era
    out["age0_x_post"] = a0 * post
    out["age0_x_year"] = a0 * (yr - 2007)
    out["age14_x_post"] = a14 * post
    out["age5p_x_post"] = a5p * post
    out["age5p_x_year"] = a5p * (yr - 2007)

    # season matters only for age>=5
    out["a5p_month_sin"] = a5p * np.sin(2 * np.pi * month / 12)
    out["a5p_month_cos"] = a5p * np.cos(2 * np.pi * month / 12)

    # within-cell age gradient (age still discriminates inside the 5+ block)
    out["age_x_a5p"] = a5p * age
    return out


V3_COLS = ["is_age0", "is_age1_4", "post2018", "age0_x_post", "age0_x_year",
           "age14_x_post", "age5p_x_post", "age5p_x_year",
           "a5p_month_sin", "a5p_month_cos", "age_x_a5p"]
