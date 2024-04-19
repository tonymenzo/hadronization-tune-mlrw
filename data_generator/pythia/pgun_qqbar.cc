// main21.cc is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Keywords: hadronization

// This is a simple test program.
// It illustrates how to feed in a single particle (including a resonance)
// or a toy parton-level configurations.

#include "Pythia8/Pythia.h"
using namespace Pythia8;

//==========================================================================

// Single-particle gun. The particle must be a colour singlet.
// Input: flavour, energy, direction (theta, phi).
// If theta < 0 then random choice over solid angle.
// Optional final argument to put particle at rest => E = m.

void fillParticle(int id, double ee, double thetaIn, double phiIn,
  Event& event, StringHistory& strings, ParticleData& pdt, Rndm& rndm, bool atRest = false,
  bool hasLifetime = false) {

  // Reset event record to allow for new event.
  event.reset();
  strings.reset();

  // Select particle mass; where relevant according to Breit-Wigner.
  double mm = pdt.mSel(id);
  double pp = sqrtpos(ee*ee - mm*mm);

  // Special case when particle is supposed to be at rest.
  if (atRest) {
    ee = mm;
    pp = 0.;
  }

  // Angles as input or uniform in solid angle.
  double cThe, sThe, phi;
  if (thetaIn >= 0.) {
    cThe = cos(thetaIn);
    sThe = sin(thetaIn);
    phi  = phiIn;
  } else {
    cThe = 2. * rndm.flat() - 1.;
    sThe = sqrtpos(1. - cThe * cThe);
    phi = 2. * M_PI * rndm.flat();
  }

  // Store the particle in the event record.
  int iNew = event.append( id, 1, 0, 0, pp * sThe * cos(phi),
    pp * sThe * sin(phi), pp * cThe, ee, mm);

  // Generate lifetime, to give decay away from primary vertex.
  if (hasLifetime) event[iNew].tau( event[iNew].tau0() * rndm.exp() );

}

//==========================================================================

// Simple method to do the filling of partons into the event record.

void fillPartons(int type, double ee, Event& event, StringHistory& strings, ParticleData& pdt,
  Rndm& rndm) {

  // Reset event record to allow for new event.
  event.reset();
  strings.reset();

  // Information on a q qbar system, to be hadronized.
  if (type == 1 || type == 12) {
    int    id = 2;
    double mm = pdt.m0(id);
    double pp = sqrtpos(ee*ee - mm*mm);
    event.append(  id, 23, 101,   0, 0., 0.,  pp, ee, mm);
    event.append( -id, 23,   0, 101, 0., 0., -pp, ee, mm);

  // Information on a g g system, to be hadronized.
  } else if (type == 2 || type == 13) {
    event.append( 21, 23, 101, 102, 0., 0.,  ee, ee);
    event.append( 21, 23, 102, 101, 0., 0., -ee, ee);

  // Information on a g g g system, to be hadronized.
  } else if (type == 3) {
    event.append( 21, 23, 101, 102,        0., 0.,        ee, ee);
    event.append( 21, 23, 102, 103,  0.8 * ee, 0., -0.6 * ee, ee);
    event.append( 21, 23, 103, 101, -0.8 * ee, 0., -0.6 * ee, ee);

  // Information on a q q q junction system, to be hadronized.
  } else if (type == 4 || type == 5) {

    // Need a colour singlet mother parton to define junction origin.
    event.append( 1000022, -21, 0, 0, 2, 4, 0, 0,
                  0., 0., 1.01 * ee, 1.01 * ee);

    // The three endpoint q q q; the minimal system.
    double rt75 = sqrt(0.75);
    event.append( 2, 23, 1, 0, 0, 0, 101, 0,
                          0., 0., 1.01 * ee, 1.01 * ee);
    event.append( 2, 23, 1, 0, 0, 0, 102, 0,
                   rt75 * ee, 0., -0.5 * ee,        ee );
    event.append( 1, 23, 1, 0, 0, 0, 103, 0,
                  -rt75 * ee, 0., -0.5 * ee,        ee );

    // Define the qqq configuration as starting point for adding gluons.
    if (type == 5) {
      int colNow[4] = {0, 101, 102, 103};
      Vec4 pQ[4];
      pQ[1] = Vec4(0., 0., 1., 0.);
      pQ[2] = Vec4( rt75, 0., -0.5, 0.);
      pQ[3] = Vec4(-rt75, 0., -0.5, 0.);

      // Minimal cos(q-g opening angle), allows more or less nasty events.
      double cosThetaMin =0.;

      // Add a few gluons (almost) at random.
      for (int nglu = 0; nglu < 5; ++nglu) {
        int iq = 1 + int( 2.99999 * rndm.flat() );
        double px, py, pz, e, prod;
        do {
          e =  ee * rndm.flat();
          double cThe = 2. * rndm.flat() - 1.;
          double phi = 2. * M_PI * rndm.flat();
          px = e * sqrt(1. - cThe*cThe) * cos(phi);
          py = e * sqrt(1. - cThe*cThe) * sin(phi);
          pz = e * cThe;
          prod = ( px * pQ[iq].px() + py * pQ[iq].py() + pz * pQ[iq].pz() )
            / e;
        } while (prod < cosThetaMin);
        int colNew = 104 + nglu;
        event.append( 21, 23, 1, 0, 0, 0, colNew, colNow[iq],
          px, py, pz, e, 0.);
        colNow[iq] = colNew;
      }
      // Update daughter range of mother.
      event[1].daughters(2, event.size() - 1);

    }

  // Information on a q q qbar qbar dijunction system, to be hadronized.
  } else if (type >= 6) {

    // The two fictitious beam remnant particles; needed for junctions.
    event.append( 2212, -12, 0, 0, 3, 5, 0, 0, 0., 0., ee, ee, 0.);
    event.append(-2212, -12, 0, 0, 6, 8, 0, 0, 0., 0., ee, ee, 0.);

    // Opening angle between "diquark" legs.
    double theta = 0.2;
    double cThe = cos(theta);
    double sThe = sin(theta);

    // Set one colour depending on whether more gluons or not.
    int acol = (type == 6) ? 103 : 106;

    // The four endpoint q q qbar qbar; the minimal system.
    // Two additional fictitious partons to make up original beams.
    event.append(  2,   23, 1, 0, 0, 0, 101, 0,
                  ee * sThe, 0.,  ee * cThe, ee, 0.);
    event.append(  1,   23, 1, 0, 0, 0, 102, 0,
                 -ee * sThe, 0.,  ee * cThe, ee, 0.);
    event.append(  2, -21, 1, 0, 0, 0, 103, 0,
                         0., 0.,  ee       , ee, 0.);
    event.append( -2,   23, 2, 0, 0, 0, 0, 104,
                  ee * sThe, 0., -ee * cThe, ee, 0.);
    event.append( -1,   23, 2, 0, 0, 0, 0, 105,
                 -ee * sThe, 0., -ee * cThe, ee, 0.);
    event.append( -2, -21, 2, 0, 0, 0, 0, acol,
                         0., 0., -ee       , ee, 0.);

    // Add extra gluons on string between junctions.
    if (type == 7) {
      event.append( 21, 23, 8, 5, 0, 0, 103, 106, 0., ee, 0., ee, 0.);
    } else if (type == 8) {
      event.append( 21, 23, 8, 5, 0, 0, 103, 108, 0., ee, 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 108, 106, 0.,-ee, 0., ee, 0.);
    } else if (type == 9) {
      event.append( 21, 23, 8, 5, 0, 0, 103, 107, 0., ee, 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 107, 108, ee, 0., 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 108, 106, 0.,-ee, 0., ee, 0.);
    } else if (type == 10) {
      event.append( 21, 23, 8, 5, 0, 0, 103, 107, 0., ee, 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 107, 108, ee, 0., 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 108, 109, 0.,-ee, 0., ee, 0.);
      event.append( 21, 23, 8, 5, 0, 0, 109, 106,-ee, 0., 0., ee, 0.);
    }

  // No more cases: done.
  }
}

//==========================================================================

int main() {

  // Loop over kind of events to generate:
  // 0 = single-particle gun.
  // 1 = q qbar.
  // 2 = g g.
  // 3 = g g g.
  // 4 = minimal q q q junction topology.
  // 5 = q q q junction topology with gluons on the strings.
  // 6 = q q qbar qbar dijunction topology, no gluons.
  // 7 - 10 = ditto, but with 1 - 4 gluons on string between junctions.
  // 11 = single-resonance gun.
  // 12 = q qbar plus parton shower.
  // 13 = g g plus parton shower.
  // It is easy to edit the line below to only study one of them.
  int    type = 1;
  
  // Set particle species and energy for single-particle gun.
  int    idGun  = (type == 0) ? 15 : 25;
  double eeGun  = (type == 0) ? 20. : 125.;
  bool   atRest = (type == 0) ? false : true;

  // The single-particle gun produces a particle at the origin, and
  // by default decays it there. When hasLifetime = true instead a
  // finite lifetime is selected and used to generate a displaced
  // decay vertex.
  bool   hasLifetime = (type == 0) ? true : false;

  // Set typical energy per parton.
  double ee = 50.0;

  // Set number of events to generate and list.
  int nEvent = 1000000;
  int nList  = 0;
  // Set a cutoff for the maximum number of rejections
  //int MAX_REJECT = 100;

  // Generator; shorthand for event and particleData.
  Pythia pythia;
  Event& event           = pythia.event;
  StringHistory& strings = pythia.strings;
  ParticleData& pdt      = pythia.particleData;

  // Set the frgamentation parameters.
  pythia.readString("StringZ:aLund = 0.68");   // Monash
  pythia.readString("StringZ:bLund = 0.98");   // Monash
  pythia.readString("StringPT:sigma = 0.335"); // Monash

  //pythia.readString("StringZ:aLund = 0.6");   // Monash
  //pythia.readString("StringZ:bLund = 1.5");   // Monash
  //pythia.readString("StringPT:sigma = 0.335"); // Monash

  // Set the quark and hadron masses.
  //pythia.readString("1:m0 = 0");
  //pythia.readString("2:m0 = 0");
  //pythia.readString("111:m0 = %f" % sqrt(0.135**2 + (1.9*(0.335/sqrt(2.)))**2));
  //pythia.readString("211:m0 = %f" % sqrt(0.135**2 + (1.9*(0.335/sqrt(2.)))**2));

  pythia.readString("ProcessLevel:all = off");
  pythia.readString("HadronLevel:Decay = off");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");
  //pythia.readString("Print:quiet = on");
  pythia.readString("StringFragmentation:TraceColours = on");

  // Set rng seed.
  pythia.readString("Random:setSeed = true");
  pythia.readString("Random:seed = 1");

  // Modify the flavor parameters such that only pions are allowed.
  pythia.readString("StringFlav:probQQtoQ = 0");
  pythia.readString("StringFlav:probStoUD = 0");
  pythia.readString("StringFlav:mesonUDvector = 0");
  pythia.readString("StringFlav:etaSup = 0");
  pythia.readString("StringFlav:etaPrimeSup = 0");
  pythia.readString("StringPT:enhancedFraction = 0");
  //pythia.readString("ParticleData:modeBreitWigner = 0")

  // Switch off automatic event listing in favour of manual.
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  // Initialize.
  pythia.init();

  // Initialize event counter.
  int eventCounter = 0;

  // Begin of event loop.
  do {
    // Set up single particle, with random direction in solid angle.
    if (type == 0 || type == 11) fillParticle( idGun, eeGun, -1., 0.,
      event, strings, pdt, pythia.rndm, atRest, hasLifetime);

    // Set up parton-level configuration.
    else fillPartons(type, ee, event, strings, pdt, pythia.rndm);

    // To have partons shower they must be set maximum allowed scale.
    // (Can be set individually to restrict radiation differently.)
    //if (type == 12 || type == 13) {
      //double scale = ee;
      //event[1].scale( scale);
      //event[2].scale( scale);

      // Now actually do the shower, for range of partons, and max scale.
      // (Most restrictive of global and individual applied to each parton.)
      //pythia.forceTimeShower( 1, 2, scale);
    //}

    // Generate events. Quit if failure.
    if (!pythia.next()) {
      cout << " Event generation aborted prematurely, owing to error!\n";
      break;
    }

    // List the string and hadron info for the first nList events.
    if (eventCounter < nList) {
      //event.list();
      strings.endA.list();
      strings.endB.list();
      strings.endC.list();
      strings.hads.list();
    }

    // Print the total number of hadrons.
    int nHadrons = strings.hads.size() - 1;
    //cout << "The number of hadrons is: " << nHadrons << endl;

    // Read in accept and reject data - reject if multiplicity != number of accepted z
    ifstream rfile("fragmentation_chain_i.txt", ios::in | ios::out);
    string line;
    int nLines = 0;
    
    // Compute the number of lines in the file
    while (getline(rfile, line)) {  // Loop through each line in the file
      nLines++;  // Incrementing line count for each line read
    }

    //cout << "The total number of lines in fragmentation_chain_i.txt: " << nLines << endl;

    // Reset the read file
    rfile.clear();
    rfile.seekg(0);

    // Append the accepted and rejected values to the master file
    if (eventCounter == 0 && nLines == nHadrons - 2) {
      // Create a new file.
      ofstream datafile("pgun_qqbar_accept_reject_z_a_0.68_b_0.98_sigma_0.335.txt");
      // Copy the event into the data file.
      while (getline(rfile, line)) {
        datafile << line << endl;
      }
      // Input a new line to separate events
      datafile << endl;
      datafile.close();

      // Output hadron level particle flow data.
      ofstream hadronfile("pgun_qqbar_hadrons_a_0.68_b_0.98_sigma_0.335.txt", ios::out | ios::app);
      for (int i = 1; i <= strings.hads.size() - 3; ++i) {
        hadronfile << strings.hads[i].px() << " " << strings.hads[i].py() << " " << strings.hads[i].pz() << " " << strings.hads[i].e() << endl;
      }
      hadronfile << endl;
      hadronfile.close();
      // Iterate the event counter.
      eventCounter++;
    } else if (nLines == nHadrons - 2) {
      // Append to the file
      ofstream datafile("pgun_qqbar_accept_reject_z_a_0.68_b_0.98_sigma_0.335.txt", ios::out | ios::in | ios::app);
      // Copy the event into the data file.
      while (getline(rfile, line)) {
        datafile << line << endl;
      }
      // Input a new line to separate events
      datafile << endl;
      datafile.close();

      // Output hadron level particle flow data.
      ofstream hadronfile("pgun_qqbar_hadrons_a_0.68_b_0.98_sigma_0.335.txt", ios::out | ios::app);
      for (int i = 1; i <= strings.hads.size() - 3; ++i) {
        hadronfile << strings.hads[i].px() << " " << strings.hads[i].py() << " " << strings.hads[i].pz() << " " << strings.hads[i].e() << endl;
      }
      hadronfile << endl;
      hadronfile.close();

      // Iterate the event counter.
      eventCounter++;
    }

    rfile.close(); // Close fragmentation_chain_i.txt

    // Clear the temporary accept-reject file
    ofstream cfile("fragmentation_chain_i.txt", ios::out | ios::trunc);
    cfile.close();

    //if (nLines == nHadrons - 2) {
    //  cout << "Huzzah!" << endl;
    //}


    // Initialize statistics.
    //Vec4 pSum = - event[0].p();
    //double chargeSum = 0.;
    //if (type == 0) chargeSum = -event[1].charge();
    //if (type == 4 || type == 5) chargeSum = -1;
    //int nFin = 0;
    //int n85 = 0;
    //int n86 = 0;
    //int n83 = 0;
    //int n84 = 0;

    // Loop over all particles.
    //for (int i = 0; i < event.size(); ++i) {
      //int status = event[i].statusAbs();
      //if (event[i].isFinal()){
        //cout << "Accepcted z value: " << abs( log(2 * event[i].pAbs() / event[i-1].e())) << endl;
      //}
      
//
    //  // Find any unrecognized particle codes.
    //  int id = event[i].id();
    //  if (id == 0 || !pdt.isParticle(id))
    //    cout << " Error! Unknown code id = " << id << "\n";
//
    //  // Find particles with E-p mismatch.
    //  double eCalc = event[i].eCalc();
    //  if (abs(eCalc/event[i].e() - 1.) > 1e-6) cout << " e mismatch, i = "
    //    << i << " e_nominal = " << event[i].e() << " e-from-p = "
    //    << eCalc << " m-from-e " << event[i].mCalc() << "\n";
//
    //  // Parton flow in event plane.
    //  if (status == 71 || status == 72) {
    //    double thetaXZ = event[i].thetaXZ();
    //    dpartondtheta.fill(thetaXZ);
    //  }
//
    //  // Origin of primary hadrons.
    //  if (status == 85) ++n85;
    //  if (status == 86) ++n86;
    //  if (status == 83) ++n83;
    //  if (status == 84) ++n84;
//
    //  // Flow of primary hadrons in the event plane.
    //  if (status > 80 && status < 90) {
    //    double eAbs = event[i].e();
    //    if (eAbs < 0.) {cout << " e < 0 line " << i; event.list();}
    //    double thetaXZ = event[i].thetaXZ();
    //    dndtheta.fill(thetaXZ);
    //    dedtheta.fill(thetaXZ, eAbs);
//
    //    // Rapidity distribution of primary hadrons.
    //    double y = event[i].y();
    //    dndySum.fill(y);
    //    if (type >= 6) {
    //      int motherId = event[event[i].mother1()].id();
    //      if (motherId > 0 ) dndyJun.fill(event[i].y());
    //      else dndyAnti.fill(event[i].y());
    //    }
      //}
//
    //  // Study final-state particles.
    //  if (event[i].isFinal()) {
    //    pSum += event[i].p();
    //    chargeSum += event[i].charge();
    //    nFin++;
    //    double pAbs = event[i].pAbs();
    //    dnparticledp.fill(pAbs);
    //  }
    //}
//
    //// Fill histograms once for each event.
    //double epDev = abs(pSum.e()) + abs(pSum.px()) + abs(pSum.py())
    //  + abs(pSum.pz());
    //epCons.fill(epDev);
    //chgCons.fill(chargeSum);
    //nFinal.fill(nFin);
    //status85.fill(n85);
    //status86.fill(n86);
    //status83.fill(n83);
    //status84.fill(n84);
    //if (epDev > 1e-3  || abs(chargeSum) > 0.1) event.list();
//
  // End of event loop.
  } while (eventCounter < nEvent);

  // Print statistics and histograms.
  //pythia.stat();
  //cout << epCons << chgCons << nFinal << dnparticledp
  //     << dndtheta << dedtheta << dpartondtheta << dndySum;
  //if (type >= 4) cout << status85 << status86 << status83
  //     << status84;
  //if (type >= 6) cout << dndyJun << dndyAnti;

}