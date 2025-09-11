import socket
import psutil
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_port_in_use(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except Exception as e:
        logger.error(f"Error checking port {port}: {str(e)}")
        return False

def terminate_process_using_port(port):
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.info['connections']
                if connections is None:
                    continue
                for conn in connections:
                    if conn.laddr.port == port:
                        logger.info(f"Found process {proc.info['name']} (PID: {proc.info['pid']}) using port {port}")
                        try:
                            proc.terminate()
                            proc.wait(timeout=5)  # Wait up to 5 seconds for termination
                            logger.info(f"Successfully terminated process {proc.info['name']} (PID: {proc.info['pid']})")
                            return True
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            logger.warning(f"Could not terminate process {proc.info['name']} (PID: {proc.info['pid']}): {str(e)}")
                            return False
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Could not access process info: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Error in terminate_process_using_port: {str(e)}")
    return False

def free_port(port):
    try:
        while is_port_in_use(port):
            if not terminate_process_using_port(port):
                logger.warning(f"No process found using port {port}, but it is still in use.")
                time.sleep(1)  # Wait a bit before retrying
    except Exception as e:
        logger.error(f"Error in free_port: {str(e)}")

if __name__ == "__main__":
    print(is_port_in_use(9000))
    free_port(9000)
    print(is_port_in_use(9000))