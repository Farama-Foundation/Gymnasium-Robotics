import os


def create_grid_cell(type_id, env_id, base_path):
    return f"""
            <a href="{base_path}{env_id}">
                <div class="env-grid__cell">
                    <div class="cell__image-container">
                        <img src="/_static/videos/{type_id}/{env_id}.gif">
                    </div>
                    <div class="cell__title">
                        <span>{' '.join(env_id.split('_')).title()}</span>
                    </div>
                </div>
            </a>
    """


def generate_page(env, limit=-1, base_path=""):
    env_type_id = env["id"]
    env_list = env["list"]
    cells = [create_grid_cell(env_type_id, env_id, base_path) for env_id in env_list]
    non_limited_page = limit == -1 or limit >= len(cells)
    if non_limited_page:
        cells = "\n".join(cells)
    else:
        cells = "\n".join(cells[:limit])

    more_btn = (
        """
<a href="./complete_list">
    <button class="more-btn">
        See More Environments
    </button>
</a>
"""
        if not non_limited_page
        else ""
    )
    return f"""
<div class="env-grid">
    {cells}
</div>
{more_btn}
    """


if __name__ == "__main__":
    """
    python gen_envs_display
    """
    # TODO: find a standard way to fetch automatically the environments
    type_dict = [
        {
            "id": "fetch",
            "list": ["FetchPickAndPlace", "FetchPush", "FetchReach", "FetchSlide",],
        },
        {"id": "hand", "list": ["HandBlock", "HandEgg", "HandPen", "HandReach",]},
        {
            "id": "hand_touch",
            "list": [
                "HandBlockTouchSensors",
                "HandEggTouchSensors",
                "HandPenTouchSensors",
            ],
        },
    ]

    for type_dict in type_dict:
        page = generate_page(type_dict)
        fp = open(
            os.path.join(
                os.path.dirname(__file__), "..", "envs", type_dict["id"], "list.html"
            ),
            "w",
            encoding="utf-8",
        )
        fp.write(page)
        fp.close()
