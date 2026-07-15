# conftest.py — block ROS launch_testing's broken pytest plugin
import sys

# Prevent broken ROS 2 launch_testing pytest plugin from loading.
# Jazzy's launch_testing requires 'lark' which is not installed in this venv.
_blocked = [
    "launch_testing.pytest.hooks",
    "launch_ros",
]
for _mod in _blocked:
    sys.modules.setdefault(_mod, type(sys)(_mod))
