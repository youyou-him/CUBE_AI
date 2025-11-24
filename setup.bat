@echo off
echo Creating project structure for CUBE...

:: ----------------------------------------------------
:: 1. Create Java package directories
:: ----------------------------------------------------
md "src\main\java\com\example\cube\controller"
md "src\main\java\com\example\cube\dto"
md "src\main\java\com\example\cube\service"
md "src\main\java\com\example\cube\model"

:: ----------------------------------------------------
:: 2. Create resource directories
:: ----------------------------------------------------
md "src\main\resources\static\css"
md "src\main\resources\static\js"
md "src\main\resources\templates\fragments"

:: ----------------------------------------------------
:: 3. Create empty Java files
:: ----------------------------------------------------
echo. > "src\main\java\com\example\cube\CubeApplication.java"
echo. > "src\main\java\com\example\cube\controller\DashboardController.java"
echo. > "src\main\java\com\example\cube\controller\CustomerController.java"
echo. > "src\main\java\com\example\cube\controller\ReviewController.java"
echo. > "src\main\java\com\example\cube\controller\MarketingController.java"
echo. > "src\main\java\com\example\cube\dto\KpiSummaryDto.java"
echo. > "src\main\java\com\example\cube\dto\CustomerSegmentDto.java"
echo. > "src\main\java\com\example\cube\service\DashboardService.java"
echo. > "src\main\java\com\example\cube\service\ApiClientService.java"

:: ----------------------------------------------------
:: 4. Create empty resource files
:: ----------------------------------------------------
echo. > "src\main\resources\static\css\style.css"
echo. > "src\main\resources\static\js\dashboard-charts.js"
echo. > "src\main\resources\templates\index.html"
echo. > "src\main\resources\templates\customer-detail.html"
echo. > "src\main\resources\templates\review-analysis.html"
echo. > "src\main\resources\templates\marketing.html"
echo. > "src\main\resources\templates\fragments\layout.html"
echo. > "src\main\resources\application.properties"

echo.
echo Project structure and files created successfully!
pause
