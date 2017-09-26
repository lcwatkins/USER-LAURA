/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(event/dihedral,ComputeEventDihedral)

#else

#ifndef LMP_COMPUTE_EVENT_DIHED_H
#define LMP_COMPUTE_EVENT_DIHED_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeEventDihedral : public Compute {
 public:
  ComputeEventDihedral(class LAMMPS *, int, char **);
  ~ComputeEventDihedral();
  void init();
  double compute_scalar();
  int region;

  // what are these
  void reset_extra_compute_fix(const char *);

 private:
  int nvalues,pflag;
  char *group2;
  int jgroup,jgroupbit,othergroupbit;

//  double compute_dihedral(int, int, int, int, int);
  double compute_dihedral(double *, double *, double *, double *);
  int find_region(double, double);
  char *id_event;
  class FixEvent *fix_event;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute event/dihedral used when dihedrals are not allowed

The atom style does not support dihedrals.

E: Invalid keyword in compute event/dihedral command

Self-explanatory.

E: No dihedral style is defined for compute event/dihedral

Self-explanatory.

E: Compute group/group group ID does not exist

Self-explanatory.

E: Could not find compute event/dihedral fix ID

Self-explanatory.

E: Compute event/dihedral has invalid fix event assigned

This is an internal LAMMPS error.  Please report it to the
developers.

E: Number of dihedrals specified in groups is not 2

Right now code requires 2 dihedrals

E: Region not defined

Something is wrong with how the phi/psi angle regions

*/
