"""Single source of truth for instruction strings (Gate P4: byte-equality).

Both the training flattener AND any eval harness must import from here so the
rendered instruction for a given state is byte-identical in train and eval.
PROMPT_WRAP is verbatim from objectnav_uninavid.py.
"""

PROMPT_WRAP = ("Imagine you are a robot programmed for navigation tasks. You have been given a "
               "video of historical observations and an image of the current observation <image>. "
               "Your assigned task is: '{}'. Analyze this series of images to determine your next "
               "four actions. The predicted action should be one of the following: forward, left, "
               "right, or stop.")

# category -> natural word used in instructions (verbatim from the eval harness)
CAT_WORD = {"chair": "chair", "bed": "bed", "plant": "plant",
            "toilet": "toilet", "tv_monitor": "TV", "sofa": "sofa"}


def collect_task(cat, n=None):
    """Collection (V2) task string; n = remaining budget in primitives, None = budget-blind."""
    word = CAT_WORD[cat]
    base = f"Find as many {word}s as possible."
    return base if n is None else base + f" You have {n} steps remaining."


def search_task(cat, n=None):
    """Find-one-and-stop (V1) task string, budget-conditioned variant of the eval 'search' style."""
    word = CAT_WORD[cat]
    base = f"Search for the {word} and stop when you reach it."
    return base if n is None else base + f" You have {n} steps remaining."


def wrap(task):
    return PROMPT_WRAP.format(task)
