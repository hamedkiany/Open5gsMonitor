# Monitor de servicios Open5GS en Docker

Script en Python para monitorizar el estado de los contenedores de **Open5GS** desplegados con **Docker** (o Docker Compose).  
Permite comprobar de forma periódica si los servicios están levantados y muestra información básica de estado por consola.

> 💡 Adapta los nombres de los contenedores y la lógica del script a tu despliegue concreto de Open5GS.

---

## Características

- Comprueba si los contenedores de Open5GS están:
  - `running`
  - `exited`
  - en reinicio o con errores
- Muestra por consola un resumen del estado de cada servicio.
- Intervalo de comprobación configurable.
- Uso muy sencillo: un solo fichero de Python.

---

## Requisitos

- Python 3.8 o superior
- Docker instalado y accesible desde el usuario que ejecuta el script
- (Opcional) Docker Compose, si usas `docker-compose.yml`
- Dependencias de Python (si tu script usa la librería oficial de Docker):

```bash
pip install docker
