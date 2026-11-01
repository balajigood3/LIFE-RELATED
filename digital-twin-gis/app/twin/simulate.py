"""Standalone simulation runner."""
from app.twin.physics import GISBayTwin
import time

def main():
    twin = GISBayTwin(dt=1.0)
    print("Starting GIS Digital Twin simulation (60 steps)...")
    twin.inject_fault("sf6_leak", severity=1.5)
    for i in range(60):
        st = twin.step()
        if i % 10 == 0:
            print(f"t={st.timestamp:5.0f}s  SF6={st.sf6_pressure_bar:.2f}bar  "
                  f"Temp={st.temp_contact_c:.1f}C  State={st.state.value}  Faults={st.active_faults}")
    print("Done.")

if __name__ == "__main__":
    main()
