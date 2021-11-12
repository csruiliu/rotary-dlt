python setup.py egg_info bdist_wheel

rm -rf build
rm -rf rotary.egg-info

pip3 uninstall -y rotary
pip3 install dist/rotary-1.0-py3-none-any.whl

rm -rf dist
