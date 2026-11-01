# 220 kV GIS Circuit Breaker – Vendor Manual Excerpt (Demo)

**Equipment**: CB-220-01 / Type: 220 kV SF6 Gas Insulated Circuit Breaker  
**Manufacturer**: GenericGIS Systems (Demo Document)

## 3.2 Alarm and Indication List

| Alarm Code | Description                        | Priority | Recommended Action                          |
|------------|------------------------------------|----------|---------------------------------------------|
| A-101      | SF6 Gas Pressure Low (Stage 1)     | High     | Investigate leak. Do not close if open.     |
| A-102      | SF6 Gas Pressure Low (Stage 2)     | Critical | Immediate isolation. Block closing.         |
| A-215      | Phase Current Imbalance > 15%      | High     | Check CT circuits, secondary wiring, load.  |
| A-301      | Spring Charge Failure              | High     | Do not operate. Check motor & limit switch. |
| A-410      | Pole Discordance                   | Critical | Trip all poles. Investigate mechanical link.|

## 4.5 Maintenance Isolation Procedure (Summary)

Before any internal inspection or gas work:

1. Open the circuit breaker remotely or locally.
2. Open the associated disconnectors (DS) on both sides.
3. Close the earthing switches (ES) on both sides of the breaker.
4. Verify zero voltage on all three phases using approved HV detector.
5. Apply personal LOTO locks on:
   - Disconnector control circuits
   - Earthing switch control circuits
   - Spring charging motor supply
   - Local/remote selector switch (set to Local + Locked)
6. For gas compartment work: reduce pressure following manufacturer procedure, verify atmospheric pressure, test for SF6 and oxygen content.
7. Only then open access covers.

**Warning**: Failure to earth the circuit can result in fatal electric shock from induced voltages or residual charge.
