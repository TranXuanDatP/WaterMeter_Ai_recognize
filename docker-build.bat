@echo off
REM Water Meter AI - Docker Build & Run Script for Windows

setlocal enabledelayedexpansion

echo ========================================
echo Water Meter AI - Docker Setup
echo ========================================
echo.

REM Check parameter
if "%1"=="" goto :usage

if /i "%1"=="build" goto :build
if /i "%1"=="run" goto :run
if /i "%1"=="stop" goto :stop
if /i "%1"=="restart" goto :restart
if /i "%1"=="logs" goto :logs
if /i "%1"=="status" goto :status
if /i "%1"=="test" goto :test
if /i "%1"=="clean" goto :clean
goto :usage

:build
echo Building Docker image...
docker-compose build 
if %errorlevel% equ 0 (
    echo [OK] Build completed
) else (
    echo [ERROR] Build failed
    exit /b 1
)
goto :end

:run
echo Starting Docker container...
docker-compose up -d
if %errorlevel% equ 0 (
    echo [OK] Container started
    echo API available at: http://localhost:8000
    echo API docs at: http://localhost:8000/docs
) else (
    echo [ERROR] Failed to start container
    exit /b 1
)
goto :end

:stop
echo Stopping Docker container...
docker-compose down
if %errorlevel% equ 0 (
    echo [OK] Container stopped
) else (
    echo [ERROR] Failed to stop container
    exit /b 1
)
goto :end

:restart
echo Restarting Docker container...
docker-compose restart
if %errorlevel% equ 0 (
    echo [OK] Container restarted
) else (
    echo [ERROR] Failed to restart container
    exit /b 1
)
goto :end

:logs
echo Showing container logs (press Ctrl+C to exit)...
docker-compose logs -f water-meter-api
goto :end

:status
echo Container status:
docker-compose ps
goto :end

:test
echo Testing API...
curl -X GET http://localhost:8000/health
if %errorlevel% neq 0 (
    echo [ERROR] API not responding
    exit /b 1
)
goto :end

:clean
echo Cleaning up Docker resources...
docker-compose down -v
docker system prune -f
echo [OK] Cleanup completed
goto :end

:usage
echo Usage: %~nx0 {build^|run^|stop^|restart^|logs^|status^|test^|clean}
echo.
echo Commands:
echo   build   - Build Docker image
echo   run     - Start container in detached mode
echo   stop    - Stop and remove container
echo   restart - Restart container
echo   logs    - View container logs
echo   status  - Show container status
echo   test    - Test API health endpoint
echo   clean   - Remove containers and unused Docker resources
echo.
echo Quick start:
echo   %~nx0 build
echo   %~nx0 run
exit /b 1

:end
endlocal
