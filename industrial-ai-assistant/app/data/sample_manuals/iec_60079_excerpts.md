# IEC 60079 Series – Key Excerpts for Hazardous Areas (Demo Corpus)

## IEC 60079-14: Electrical Installations Design, Selection and Erection

### 5.3 Equipment Selection for Zone 1
Electrical equipment installed in Zone 1 shall be selected from equipment certified for Zone 0 or Zone 1, or from equipment with a type of protection suitable for Zone 1 (e.g. Ex d, Ex e, Ex i, Ex p, Ex m).

### 5.4 Equipment Selection for Zone 2
Equipment suitable for Zone 0 or Zone 1 may be used. In addition, Ex n equipment and equipment with protection level EPL Gc is acceptable.

### 9.3 Isolation
Means shall be provided to isolate all live conductors, including the neutral, from the supply. Isolation shall be arranged so that it can be locked in the open position.

**Critical Note**: Isolation switches for equipment in hazardous areas shall themselves be located outside the hazardous area wherever practicable, or shall be of a suitable explosion-protected type.

## IEC 60079-17: Electrical Installations Inspection and Maintenance

### 4.3 Competence of Personnel
Inspection and maintenance of installations in hazardous areas shall only be carried out by experienced personnel whose training has included instruction on the various types of protection and installation practices, relevant rules and regulations, and general principles of area classification.

### 5.2 Isolation before Maintenance
Before any work is carried out on electrical equipment in a hazardous area, the equipment shall be isolated from all sources of supply (including neutral and any control or monitoring circuits) and measures taken to prevent re-energization. This isolation shall be verified by testing.

### 5.3 Live Working
Live working in hazardous areas is generally prohibited. Where live working is unavoidable, a formal Permit-to-Work system with additional safeguards is mandatory.

## IEC 60079-19: Equipment Repair, Overhaul and Reclamation

### 4.2 Competence
Repair and overhaul shall only be carried out by competent persons with documented training specific to the type of protection involved.

### 5.1 Documentation
All repairs shall be fully documented, including the identity of the equipment, nature of the work, and any changes to the original certification.

## SF6 Gas Insulated Switchgear – General Safety (Vendor-Agnostic Best Practice)

### SF6 Handling
- SF6 is a greenhouse gas and can displace oxygen in confined spaces, creating an asphyxiation hazard.
- Before opening any gas compartment, verify pressure is at atmospheric or follow manufacturer purging procedure.
- Use SF6 detectors in the work area. Never enter a compartment without confirming breathable atmosphere.
- Arc by-products of SF6 (especially after internal arcing) can be toxic. Special PPE and neutralization procedures are required.

### Pressure Alarms
- Low SF6 pressure alarm typically indicates leakage. Do not operate the breaker under low pressure if interlocks permit.
- Investigate and repair leaks only after proper isolation, locking, and atmospheric testing.

## Typical LOTO Principles for HV GIS (Aligned with OSHA 1910.147 & IEC practice)

1. Prepare – Identify all energy sources (electrical primary, secondary, control, hydraulic, spring energy, SF6 pressure).
2. Shut down – Normal stop of equipment.
3. Isolate – Open disconnectors / isolators, withdraw truck if applicable, close earthing switches.
4. Lock & Tag – Apply personal locks and tags on all isolation points.
5. Release residual energy – Discharge capacitors, release spring energy, vent residual pressure if safe.
6. Verify – Try-start / voltage test / visual confirmation of open contacts and earthed state.
7. Perform work.
8. Release from LOTO – Reverse sequence, only by the person who applied the lock (or under formal transfer procedure).
