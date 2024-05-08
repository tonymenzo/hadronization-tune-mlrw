// main301.cc is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Christian Bierlich <christian.bierlich@fysik.lu.se>
//          Stephen Mrenna <mrenna@fnal.gov>
//          Philip Ilten <philten@cern.ch
// Modified by: Tony Menzo <menzoad@mail.uc.edu>

// Keywords: hadronization; reweighting; tuning; parallelism; matplotlib

// Performs a scan over the Lund a and b parameters to understand the 
// allowed "length-scales" for reweighting 

// Pythia includes.
#include "Pythia8/Pythia.h"

using namespace Pythia8;

int main() {

  // Choose to reweight kinematic (0) or flavor (1) hadronization
  // parameters. Both can be reweighted simultaneously, but the
  // separation is kept here for illustrative purposes.
  //int type = 0;

  // Number of events to generate per run.
  int nEvent = 1e6;

  float aLundScanArray[] = { 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8 };
  float bLundScanArray[] = { 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8 };
  //int nScan = int(end(aLundScanArray)-begin(aLundScanArray)) * int(end(bLundScanArray)-begin(bLundScanArray));

  // Define the new set of kinematic parameters that we wish to reweight to.
  //double aLund   = 0.6;  // StringZ:aLund, default 0.68
  //double bLund   = 0.9;  // StringZ:bLund, default 0.98
  //double rFactC  = 1.3;  // StringZ:rFactC, default 1.32
  //double rFactB  = 0.9;  // StringZ:rFactB, default 0.855
  //double ptSigma = 0.3;  // StringPT:sigma, default 0.335

  // Define the new set of flavor parameters that we wish to reweight to.
  //double rho = 0.2;  // StringFlav:ProbStoUD, default 0.217
  //double xi  = 0.1;  // StringFlav:ProbQQtoQ, default 0.081
  //double x   = 0.9;  // StringFlav:ProbSQtoQQ, default 0.915
  //double y   = 0.04; // StringFlav:ProbQQ1toQQ0, default 0.0275

  // Create and configure Pythia.
  Pythia pythia;
  Event& event      = pythia.event;
  ParticleData& pdt = pythia.particleData;

  // Setup Pythia settings.
  pythia.readString("StringZ:aLund = 0.5"); 
  pythia.readString("StringZ:bLund = 0.45"); 

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

  // Create variations string
  // Syntax: Groups are separated by commas, elements within a group are separated by spaces.
  // Each group must have a unique name. For example:
  //
  // {<group_name_1> <parameter_1>=<value_1> <parameters_2>=<value_2> ... ,
  //  <group_name_2> <parameter_1>=<value_1> <parameters_2>=<value_2> ... ,
  //                              .
  //                              .
  //                              .
  //  <group_name_N> <parameter_1>=<value_1> <parameters_2>=<value_2> ... }
  string variationsString = "VariationFrag:List = {";
  for (int i = 0; i < int(end(aLundScanArray)-begin(aLundScanArray)); i++)
  {
      for (int j = 0; j < int(end(bLundScanArray)-begin(bLundScanArray)); j++)
      {
        // Append the group with the name specifying the parameter variation
        variationsString.append("kineVar_a_" + to_string(aLundScanArray[i]) + "_b_" + to_string(bLundScanArray[j]) +
        " frag:aLund=" + to_string(aLundScanArray[i]) + 
        " frag:bLund=" + to_string(bLundScanArray[j]) + ",");
      }
  }
  variationsString.pop_back();
  variationsString.pop_back();
  variationsString.append("}");

  //cout << variationsString << endl;

  // Feed the variations to Pythia.
  pythia.readString(variationsString);

  // Configure the in-situ kinematic or flavor reweighting.
  //if (type == 0) {
  //  pythia.readString("VariationFrag:List = {kineVar frag:aLund=" + to_string(aLund) + 
  //    " frag:bLund=" + to_string(bLund) + 
  //    //" frag:rFactC=" + to_string(rFactC) + 
  //    //" frag:rFactB=" + to_string(rFactB) +
  //    //" frag:ptSigma=" + to_string(ptSigma) + 
  //    "}");
  //} else if (type == 1) {
  //  pythia.readString("VariationFrag:List = {flavVar frag:rho="
  //    + to_string(rho) + " frag:xi=" + to_string(xi) + " frag:x="
  //    + to_string(x) + " frag:y=" + to_string(y) + "}");
  //  pythia.readString("StringFlav:popcornRate = 0");
  //  pythia.readString("HadronLevel:Decay = off");
  //}

  // Initialize Pythia.
  pythia.init();

  // Define weight names
  vector<string> names = {};

  for (int i = 0; i < int(end(aLundScanArray)-begin(aLundScanArray)); i++)
  {
      for (int j = 0; j < int(end(bLundScanArray)-begin(bLundScanArray)); j++)
      {
        names.push_back("kineVar_a_" + to_string(aLundScanArray[i]) + "_b_" + to_string(bLundScanArray[j]));
      }
  }

  //for (string i: names)
  //  cout << i << ' ';

  // Track the weights.
  map<string, double> wgts, sumWgts, sumWgt2s;
  for (string &name : names)
    wgts[name] = sumWgts[name] = sumWgt2s[name] = 0;
  //names.pop_back();

  // Define a q-qbar string configuration
  double ee = 50.0;
  int    id = 2;
  double mm = pdt.m0(id);
  double pp = sqrtpos(ee*ee - mm*mm);

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
    //wgts["default"] = 1;

    // The weight given by the in-situ reweighting.
    //wgts["insitu"] = pythia.info.getGroupWeight(0);
    //cout << pythia.info.getGroupWeight(0) << endl;
    //cout << pythia.info.getGroupWeight(10) << endl;
    //cout << pythia.info.getGroupName(10) << endl;

    // Keep track of the weights for diagnostics.
    int varCounter = 0;
    for (string &name : names) {
      //cout << pythia.info.getGroupWeight(varCounter) << endl;
      wgts[name] = pythia.info.getGroupWeight(varCounter);
      sumWgts[name]  += wgts[name];
      sumWgt2s[name] += pow2(wgts[name]);
      varCounter+=1;
    }
    //cout << varCounter << endl;

    // Write out weight data.
    //ofstream weightfile(weightPATH, ios::out | ios::app);
    //weightfile << wgts["insitu"] << endl;
    //weightfile.close();

    // Iterate the event counter.
    eventCounter += 1;
  } while (eventCounter < nEvent);

  // Write out weight data.
  string weightMetricsPATH = "./weight_metrics_base_a_0.5_b_0.45_n_1e6.dat";
  // Open the metrics file
  ofstream weightmetricsfile(weightMetricsPATH, ios::out | ios::app);
  // Print the reweighting stats.
  // The 1 - mu should be statistically consistent with zero if the
  // reweighting has proper coverage.
  // The n_eff gives the statistical power of the reweighted sample.

  for (string &name : names) {
    double w(sumWgts[name]), w2(sumWgt2s[name]), n(nEvent);
    cout << name << "\n"
         << "\t1 - mu = " << scientific << setprecision(3) << abs(1. - w/n)
         << " +- "<< abs(1. - sqrt((w2/n - pow2(w/n))*n/(n - 1)))/sqrt(n)
         << "\n\tn_eff  = " << scientific << setprecision(3) << w*w/(n*w2)
         << "\n";
    // Write out to the metrics file
    weightmetricsfile << name << endl;
    weightmetricsfile << scientific << setprecision(3) << abs(1. - w/n)
         << " " << abs(1. - sqrt((w2/n - pow2(w/n))*n/(n - 1)))/sqrt(n) 
         << " " << scientific << setprecision(3) << w*w/(n*w2) << endl;
    weightmetricsfile << endl;
  }
  weightmetricsfile.close();
}