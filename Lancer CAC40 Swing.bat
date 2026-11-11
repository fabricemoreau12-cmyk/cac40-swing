@echo off
rem Ouvre CAC40_Swing.html en mode application (sans barre d'adresse) dans Microsoft Edge
set "APP=%~dp0CAC40_Swing.html"
set "APP=%APP:\=/%"
start "" msedge --app="file:///%APP%" --window-size=1450,950
