#!/usr/bin/env bash

# Moverse al directorio del script para que las rutas ../ funcionen
cd "$(dirname "$0")"

# Ejecutar el gNB en segundo plano SIN mostrar salida y sin bloquear
sudo  srsue ue2_zmq.conf > /dev/null 2>&1 &
