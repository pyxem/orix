@echo off
set root_dir=%~dp0
pushd %root_dir%
for %%G in (base grid io plot quaternion scalar tests vector) do (
	cd %%G
	for /R %%G in ("*.py") do autopep8 --aggressive --in-place --max-line-length 130 %%G
	cd ..
)
popd
