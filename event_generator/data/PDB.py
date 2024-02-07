"""
# pdb.py is a part of the MLHAD package.
# Copyright (C) 2022 MLHAD authors (see AUTHORS for details).
# MLHAD is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

class ParticleData:
    """
    The 'ParticleData' class stores all the necessary information to
    define a particle.
    """
    
    def __init__(self, pid = None, name = None, mass = None, tau = None,
                 spin = None, charge = None, colour = None):
        """
        Initialize the class with the following: 'pid' is the particle ID
        number, 'name' the name, 'mass' the mass in GeV, 'tau' the
        proper lifetime in mm/c, 'spin' the particle spin, 'charge' is
        three times the electromagnetic charge, and 'colour' is the
        colour type.
        """
        self.pid = pid
        self.name = name
        self.mass = mass
        self.tau = tau
        self.spin = spin
        self.charge = charge
        self.colour = colour
        self.anti = None
    
    def __str__(self):
        """
        Return a string to print of this particle data.
        """
        return ("%6s: %s\n"*6 + "%6s: %s") % (
            "pid", self.pid, "name", self.name, "mass", self.mass,
            "tau", self.tau, "spin", self.spin, "charge", self.charge,
            "colour", self.colour)
    def __repr__(self):
        """
        Return the representation of this particle data.
        """
        return "ParticleData(%r, %r, %r, %r, %r, %r, %r)" % (
            self.pid, self.name, self.mass, self.tau, self.spin,
            self.charge, self.colour)
    
class ParticleDatabase(dict):
    """
    The 'ParticleDatabase' initializes and stores the 'ParticleData' for
    all particle in the 'ParticleData.xml' file from Pythia 8.
    """
    
    def __init__(self, xmlfile = "data/ParticleData.xml"):
        """
        Read in the particle data from the XML file 'xmlfile'.
        """
        # Instantiate the base class.
        dict.__init__(self)
        # Open the XML file.
        xml = open(xmlfile)
        # Create the particle string.
        pstr = ""
        # Create the list of particle strings.
        pstrs = []
        # Loop over the file.
        for line in xml:
            line = line.strip()
            if line.startswith("<particle"): pstr = line
            elif pstr and line.endswith(">"):
                self.add(pstr + " " + line)
                pstr = ""
        xml.close()
    
    def add(self, pstr):
        """
        Parses the XML for a particle and adds it to the database.
        """
        import shlex
        # Create the default dictionary.
        pdct = {"id": 0, "name": "", "antiName": None, "spinType": 0,
                "chargeType": 0, "colType": 0, "m0": 0, "tau0": 0}
        # Split the string by spaces, and loop over the entries.
        for pair in shlex.split(pstr[9:-1]):
            # Split each string into a key-value pair.
            key, val = pair.split("=", 1)
            pdct[key] = val
        # Add the particle.
        pdat = ParticleData(
            int(pdct["id"]), pdct["name"], float(pdct["m0"]),
            float(pdct["tau0"]), int(pdct["spinType"]),
            int(pdct["chargeType"]), int(pdct["colType"]))
        self[pdat.pid] = pdat
        self[pdat.name] = pdat
        # Add the anti-particle if it exists, flip PID and charge.
        if pdct["antiName"]:
            adat = ParticleData(
            -int(pdct["id"]), pdct["antiName"], float(pdct["m0"]),
            float(pdct["tau0"]), int(pdct["spinType"]),
            -1*int(pdct["chargeType"]), int(pdct["colType"]))
            self[adat.pid] = adat
            self[adat.name] = adat
            pdat.anti = adat
            adat.anti = pdat

if __name__ == "__main__":
    # Initialize a pdb class object
    pdb = ParticleDatabase()
    print(pdb[1]) # Equivalent to pdb['d']