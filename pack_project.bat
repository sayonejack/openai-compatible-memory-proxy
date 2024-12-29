@echo off
setlocal enabledelayedexpansion

:: 设置时间戳
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set datetime=%datetime:~0,8%_%datetime:~8,6%

:: 获取当前目录名
for %%f in ("%CD%") do set dirname=%%~nxf

:: 设置输出文件名
set OUTPUT_ZIP=%dirname%_%datetime%.zip

:: 创建临时文件存储排除规则
set EXCLUDE_FILE=exclude_rules.txt

:: 确保 backups 目录存在
if not exist "backups" mkdir "backups"

:: 从.gitignore生成排除规则
echo Creating exclude rules from .gitignore...
type nul > %EXCLUDE_FILE%
for /f "tokens=*" %%a in (.gitignore) do (
    set "line=%%a"
    :: 跳过空行和注释行
    if "!line:~0,1!" neq "#" if "!line!" neq "" (
        :: 处理目录路径
        if "!line:~-1!" == "/" (
            :: 如果是目录（以/结尾），添加目录本身和其内容
            echo !line!** >> %EXCLUDE_FILE%
            echo !line! >> %EXCLUDE_FILE%
        ) else (
            :: 处理通配符模式
            if "!line:~0,2!" == "**" (
                :: 如果以**开头，保持原样
                echo !line! >> %EXCLUDE_FILE%
            ) else if "!line:~0,1!" == "*" (
                :: 如果以*开头，确保能匹配任意路径
                echo !line! >> %EXCLUDE_FILE%
                echo **/!line! >> %EXCLUDE_FILE%
            ) else (
                :: 其他情况，添加原始规则和包含路径的版本
                echo !line! >> %EXCLUDE_FILE%
                if not "!line:~0,1!" == "/" (
                    echo **/!line! >> %EXCLUDE_FILE%
                )
            )
        )
        :: 为每个非通配符条目添加递归版本
        if not "!line:~0,1!" == "*" if not "!line:~-1!" == "/" (
            echo **/*!line! >> %EXCLUDE_FILE%
        )
    )
)

:: 添加额外的基本排除规则
echo %EXCLUDE_FILE% >> %EXCLUDE_FILE%
echo %OUTPUT_ZIP% >> %EXCLUDE_FILE%
echo backups\%OUTPUT_ZIP% >> %EXCLUDE_FILE%
echo pack_project.bat >> %EXCLUDE_FILE%
echo backups\** >> %EXCLUDE_FILE%

:: 检查是否安装了7-Zip
where 7z >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到7-Zip。请安装7-Zip并确保其在系统PATH中。
    echo 下载地址: https://www.7-zip.org/
    goto :cleanup
)

:: 创建ZIP文件
echo Creating backups\%OUTPUT_ZIP%...
7z a -tzip backups\%OUTPUT_ZIP% * -xr@%EXCLUDE_FILE%

:: 检查打包结果
if %ERRORLEVEL% equ 0 (
    echo 打包成功！输出文件: backups\%OUTPUT_ZIP%
) else (
    echo 打包失败！
)

:cleanup
:: 清理临时文件
if exist %EXCLUDE_FILE% del %EXCLUDE_FILE%

pause