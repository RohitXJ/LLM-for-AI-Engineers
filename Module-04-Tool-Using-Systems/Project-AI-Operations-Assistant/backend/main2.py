from pathlib import Path
import logging
from core.manager import Manager
from core.engine.agent import Agent

logging.basicConfig(
    filename='logs/ai.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
#Globals
main_logger = logging.getLogger("MainApp")
LOG_ADD = r"alerts"

def main():
    # 1. Initialize System
    manager = Manager(main_logger)
    manager.engineStartup()

    # 2. Instantiate Agent
    agent = Agent(manager)

    # 3. Process Alerts
    alerts = manager.read_alerts(LOG_ADD)
    print(f"Loaded {len(alerts)} alerts.")

    for alert in alerts:
        # Only process if not already resolved
        if manager.states.get(alert.alert_id) != "RESOLVED":
            agent.process_alert(alert)
        else:
            print(f"Skipping alert {alert.alert_id} - already resolved.")

if __name__ == "__main__":
    main()