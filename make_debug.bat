@ECHO OFF


@REM change the current directory to the directory where the batch file itself is located
pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=docs/source
set BUILDDIR=docs
if "%SPHINXOPTS%"=="" (
    set SPHINXOPTS=-vvv
)

@REM redirecting the output (both standard output and standard error) of the sphinx-build command to the null device, effectively suppressing any output or error messages.
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

@REM checks if the first command-line argument (%1) is empty or not.
@REM    If %1 is empty, jumps to 'help'.
if "%1" == "" goto help


@REM sphinx-build: runs a specific build target (given by %1).
@REM -M: specifies a "make-mode" command, which allows you to run
@REM 			commands like html, clean, etc.
@REM %1: The target to build (e.g., html).
@REM %SOURCEDIR%: The source directory (docs/source).
@REM %BUILDDIR%: 	The build directory (docs).
@REM %SPHINXOPTS% and %O%: Any additional options.

@REM echo %SPHINXBUILD%
@REM echo %1
@REM echo %SOURCEDIR%
@REM echo %BUILDDIR%
@REM echo %SPHINXOPTS%
@REM echo %O%

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O% > build.log 2>&1


REM if exist %BUILDDIR%\html (
REM Move contents of doc/html/ to doc/ (including subfolders)
REM Delete the html folder
@REM move /Y %BUILDDIR%\html\* %BUILDDIR%\
xcopy /E /Y %BUILDDIR%\html\* %BUILDDIR%\ >NUL 2>&1
rmdir /S /Q %BUILDDIR%\html
@REM ) else (
@REM 	echo html file not exists
@REM )

goto end


:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
