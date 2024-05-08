// main301.cc is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Christian Bierlich <christian.bierlich@fysik.lu.se>
//          Stephen Mrenna <mrenna@fnal.gov>
//          Philip Ilten <philten@cern.ch
// Modified by: Tony Menzo <menzoad@mail.uc.edu>

// Keywords: hadronization; reweighting; tuning; parallelism; matplotlib

// Demonstrates how to reweight an event for different kinematic or flavor
// hadronization parameters using in-situ reweighting.

// Pythia includes.
#include "Pythia8/Pythia.h"

using namespace Pythia8;

int main() {

  // Choose to reweight kinematic (0) or flavor (1) hadronization
  // parameters. Both can be reweighted simultaneously, but the
  // separation is kept here for illustrative purposes.
  int type = 0;

  // Number of events to generate per run.
  int nEvent = 1e5;

  // Define the new set of kinematic parameters that we wish to reweight to.
  double aLund   = 0.6;  // StringZ:aLund, default 0.68
  double bLund   = 0.9;  // StringZ:bLund, default 0.98
  //double rFactC  = 1.3;  // StringZ:rFactC, default 1.32
  //double rFactB  = 0.9;  // StringZ:rFactB, default 0.855
  //double ptSigma = 0.3;  // StringPT:sigma, default 0.335

  // Define the new set of flavor parameters that we wish to reweight to.
  double rho = 0.2;  // StringFlav:ProbStoUD, default 0.217
  double xi  = 0.1;  // StringFlav:ProbQQtoQ, default 0.081
  double x   = 0.9;  // StringFlav:ProbSQtoQQ, default 0.915
  double y   = 0.04; // StringFlav:ProbQQ1toQQ0, default 0.0275

  // Create and configure Pythia.
  Pythia pythia;
  Event& event           = pythia.event;
  //StringHistory& strings = pythia.strings;
  ParticleData& pdt      = pythia.particleData;

  // Setup Pythia settings.
  pythia.readString("ProcessLevel:all = off");
  pythia.readString("HadronLevel:Decay = off");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");
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

  // Switch off automatic event listing in favour of manual.
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");

  // Configure the in-situ kinematic or flavor reweighting.
  if (type == 0) {
    pythia.readString("VariationFrag:List = {kineVar frag:aLund=" + to_string(aLund) + 
      " frag:bLund=" + to_string(bLund) + 
      //" frag:rFactC=" + to_string(rFactC) + 
      //" frag:rFactB=" + to_string(rFactB) +
      //" frag:ptSigma=" + to_string(ptSigma) + 
      "}");
  } else if (type == 1) {
    pythia.readString("VariationFrag:List = {flavVar frag:rho="
      + to_string(rho) + " frag:xi=" + to_string(xi) + " frag:x="
      + to_string(x) + " frag:y=" + to_string(y) + "}");
    pythia.readString("StringFlav:popcornRate = 0");
    pythia.readString("HadronLevel:Decay = off");
  }

  // Initialize Pythia.
  pythia.init();

  // Identified final state hadrons to include in the flavor histograms.
  vector<int> hadrons = {
    111, 211, 221, 331, 213, 223, 321, 311, 333, 2212, 2112, 2214, 2224, 3222,
    3212, 3122, 3322, 3334};

  // Define multiplicity histograms. For kinematics, we look at
  // charged multiplicity while for flavor we look at multiplicity per
  // species.
  // default: the default parameters in Pythia
  // insitu:  in-situ reweighted
  // rerun:   a run with the varied parameters
  vector<string> names = {"default", "insitu", "rerun"};
  map<string, Hist> hists;
  for (string &name : names) {
    if (type == 0)
      hists[name] = Hist(name, 25, 2, 51);
    else if (type == 1)
      hists[name] = Hist(name, hadrons.size(), 0, hadrons.size());
  }

  // Track the weights.
  map<string, double> wgts, sumWgts, sumWgt2s;
  for (string &name : names)
    wgts[name] = sumWgts[name] = sumWgt2s[name] = 0;
  names.pop_back();

  // Define a q-qbar string configuration
  double ee = 50.0;
  int    id = 2;
  double mm = pdt.m0(id);
  double pp = sqrtpos(ee*ee - mm*mm);

  // Set data paths.
  string basePATH = "generated_data/pgun_qqbar_hadrons_a_0.68_b_0.98.dat";
  string weightPATH = "generated_data/pgun_qqbar_weights_a_0.68_0.6_b_0.98_0.9.dat";
  string pertPATH = "generated_data/pgun_qqbar_hadrons_a_0.6_b_0.9.dat";

  // Run events.
  int eventCounter = 0;
  do {
    // Reset event record to allow for new event.
    event.reset();

    // Append new event to the event record
    event.append(  id, 23, 101,   0, 0., 0.,  pp, ee, mm);
    event.append( -id, 23,   0, 101, 0., 0., -pp, ee, mm);

    // Generate events. Quit if failure.
    if (!pythia.next()) {
      cout << "Event generation aborted prematurely, owing to error!\n";
      break;
    }

    // List the event.
    //event.list();

    // For the default parameters, the weight is just 1.
    wgts["default"] = 1;

    // The weight given by the in-situ reweighting.
    wgts["insitu"] = pythia.info.getGroupWeight(0);

    // Keep track of the weights for diagnostics.
    for (string &name : names) {
      sumWgts[name]  += wgts[name];
      sumWgt2s[name] += pow2(wgts[name]);
    }

    // Write out hadron data.
    ofstream basehadronfile(basePATH, ios::out | ios::app);
    for (int i = 3; i <= event.size() - 1; ++i) {
      // Convention (px, py, pz, E, m)
      basehadronfile << event[i].px() << " " << event[i].py() << " " << event[i].pz() << " " << event[i].e() << " " << event[i].m() << endl;
    }
    basehadronfile << endl;
    basehadronfile.close();

    // Write out weight data.
    ofstream weightfile(weightPATH, ios::out | ios::app);
    weightfile << wgts["insitu"] << endl;
    weightfile.close();

    // Iterate the event counter.
    eventCounter += 1;
  } while (eventCounter < nEvent);

  // Fill the histograms.
  //  int mult = 0;
  //  for (const Particle &prt : pythiaPtr->event) {
  //    if (!prt.isFinal()) continue;
  //    if (type == 0) {
  //      if (prt.isCharged()) ++mult;
  //    } else if (type == 1) {
  //      int pid = prt.idAbs();
  //      int idx = -1;
  //      for (int iHad = 0; iHad < (int)hadrons.size(); ++iHad)
  //        if (pid == hadrons[iHad]) {idx = iHad; break;}
  //      if (idx < 0) continue;
  //      for (string &name : names) hists[name].fill(idx, wgts[name]);
  //    }
  //  }
  //  if (type == 0)
  //    for (string &name : names) hists[name].fill(mult, wgts[name]);
  //});
  //pythia.stat();

  // Rerun Pythia with the varied parameters.
  if (type == 0) {
    pythia.settings.parm("StringZ:aLund",  aLund);
    pythia.settings.parm("StringZ:bLund",  bLund);
    //pythia.settings.parm("StringZ:rFactC", rFactC);
    //pythia.settings.parm("StringZ:rFactB", rFactB);
    //pythia.settings.parm("StringPT:sigma", ptSigma);
  } else if (type == 1) {
    pythia.settings.parm("StringFlav:ProbStoUD",    rho);
    pythia.settings.parm("StringFlav:ProbQQtoQ",    xi);
    pythia.settings.parm("StringFlav:ProbSQtoQQ",   x);
    pythia.settings.parm("StringFlav:ProbQQ1toQQ0", y);
  }
  pythia.settings.wvec("VariationFrag:List", {});
  pythia.init();

  // Run events.
  eventCounter = 0;
  do {
    // Reset event record to allow for new event.
    event.reset();

    // Apppend a new event to the event record.
    event.append(  id, 23, 101,   0, 0., 0.,  pp, ee, mm);
    event.append( -id, 23,   0, 101, 0., 0., -pp, ee, mm);

    // Generate events. Quit if failure.
    if (!pythia.next()) {
      cout << "Event generation aborted prematurely, owing to error!\n";
      break;
    }

    // Write out hadron data.
    ofstream perthadronfile(pertPATH, ios::out | ios::app);
    for (int i = 3; i <= event.size() - 1; ++i) {
      // Convention (px, py, pz, E, m)
      perthadronfile << event[i].px() << " " << event[i].py() << " " << event[i].pz() << " " << event[i].e() << " " << event[i].m() << endl;
    }
    perthadronfile << endl;
    perthadronfile.close();

    // Keep track of the weights.
    sumWgts["rerun"]  += 1;
    sumWgt2s["rerun"] += 1;

    // Output the hadron info and weights
    eventCounter += 1;
  } while (eventCounter < nEvent);

  //pythia.run( nEvent, [&](Pythia* pythiaPtr) {
  //  sumWgts["rerun"]  += 1;
  //  sumWgt2s["rerun"] += 1;
  //  int mult = 0;
  //  for (const Particle &prt : pythiaPtr->event) {
  //    if (!prt.isFinal()) continue;
  //    if (type == 0) {
  //      if (prt.isCharged()) ++mult;
  //    } else if (type == 1) {
  //      int pid = prt.idAbs();
  //      int idx = -1;
  //      for (int iHad = 0; iHad < (int)hadrons.size(); ++iHad)
  //        if (pid == hadrons[iHad]) {idx = iHad; break;}
  //      if (idx >= 0) hists["rerun"].fill(idx, 1.);
  //    }
  //  }
  //  if (type == 0) hists["rerun"].fill(mult, 1);
  //});
  //pythia.stat();

  // Normalize the histograms.
  //for (auto &hist : hists) hist.second /= sumWgts[hist.first];

  // Print the histogram ratios.
  //string xlabel;
  //if (type == 0) {
  //  xlabel = "multiplicity";
  //} else if (type == 1) {
  //  for (int iHad = 0; iHad < (int)hadrons.size(); ++iHad) {
  //    string name = pythia.particleData.name(hadrons[iHad]);
  //    cout << left << setw(3) << iHad << ": " << name << "\n";
  //    xlabel += " " + name + "(" + to_string(iHad) + ")";
  //  }
  //}
  //for (auto &hist : hists)
  //  cout << "\n" << hist.first << hist.second/hists["default"];

  // Print the reweighting stats.
  // The 1 - mu should be statistically consistent with zero if the
  // reweighting has proper coverage.
  // The n_eff gives the statistical power of the reweighted sample.
  for (string &name : names) {
    double w(sumWgts[name]), w2(sumWgt2s[name]), n(sumWgts["default"]);
    cout << name << "\n"
         << "\t1 - mu = " << scientific << setprecision(3) << abs(1. - w/n)
         << " +- "<< abs(1. - sqrt((w2/n - pow2(w/n))*n/(n - 1)))/sqrt(n)
         << "\n\tn_eff  = " << scientific << setprecision(3) << w*w/(n*w2)
         << "\n";
  }

  // Create the Python plot and return.
  //HistPlot hpl("main301plot");
  //hpl.frame("main301plot", title, xlabel, "n(variation)/n(default)");
  //for (string &name : names)
  //  hpl.add(hists[name]/hists["default"], "e", name);
  //hpl.add(hists["rerun"]/hists["default"], "e", "rerun");
  //hpl.plot();
  //return 0;

}
