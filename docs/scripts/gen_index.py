__author__ = "Feng Gu"
__email__ = "contact@fenggu.me"

"""
   isort:skip_file
"""

import os

readme_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "README.md",
)

output_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "index.md",
)


all_text = """---
hide-toc: true
firstpage:
lastpage:
---\n"""

index_toctree = """
```{toctree}
:hidden:
:caption: Environments
envs/index
```
```{toctree}
:hidden:
:caption: Development
Github <https://github.com/Farama-Foundation/Gymnasium-Robotics>
Donate <https://farama.org/donations>
Contribute to the Docs <https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/.github/PULL_REQUEST_TEMPLATE.md>
```
"""
# gen index.md
with open(readme_path, "r") as f:
    readme = f.read()

    """
    sections = [precommit, img, main, fetch, img, hand, image, sensor, image, citation]
    """
    sections = readme.split("<br>")
    all_text += index_toctree
    all_text += sections[2]


with open(output_path, "w") as f:
    f.write(all_text)
