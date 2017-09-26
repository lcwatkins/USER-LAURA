/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include <string.h>
#include "compute_event_dihed.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "update.h"
#include "domain.h"
#include "force.h"
#include "dihedral.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "fix_event.h"
#include "modify.h"
#include "universe.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define DELTA 10000
#define SMALL 0.001
#define INVOKED_SCALAR 1

/* ---------------------------------------------------------------------- */

ComputeEventDihedral::ComputeEventDihedral(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), id_event(NULL), fix_event(NULL), group2(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal compute event/dihedral command");

  if (atom->avec->dihedrals_allow == 0)
    error->all(FLERR,
               "Compute event/dihedral used when dihedrals are not allowed");

  local_flag = 1;
  nvalues = narg - 3;
  // do I want to keep this? 
  if (nvalues == 1) size_local_cols = 0;
  else size_local_cols = nvalues;

  pflag = -1;
  nvalues = 0;

  // get rid of this argument?
  /*  
  for (int iarg = 3; iarg < narg; iarg++) {
    if (strcmp(arg[iarg],"phi") == 0) pflag = nvalues++;
    else error->all(FLERR,"Invalid keyword in compute event/dihedral command");
  }
  */

  scalar_flag = 1;
  extscalar = 0;

  // Get other group for 2nd dihedral to compute
  int n = strlen(arg[3]) + 1;
  group2 = new char[n];
  strcpy(group2,arg[3]);

  jgroup = group->find(group2);
  if (jgroup == -1) 
    error->all(FLERR,"Compute event/dihedral group ID does not exist");
  jgroupbit = group->bitmask[jgroup];

  // fix event ID will be set later by accelerated dynamics method

  id_event = NULL;

}

/* ---------------------------------------------------------------------- */

ComputeEventDihedral::~ComputeEventDihedral()
{
  delete [] id_event;
  delete [] group2;
}

/* ---------------------------------------------------------------------- */

void ComputeEventDihedral::init()
{
  if (force->dihedral == NULL)
    error->all(FLERR,"No dihedral style is defined for compute event/dihedral");

  // if id_event is not set, this compute is not active
  // if set by PRD, then find fix which stores original atom coords
  // check if it is correct style
 
  if (id_event != NULL) {
    int ifix = modify->find_fix(id_event);
    if (ifix < 0) error->all(FLERR,
                             "Could not find compute event/dihedral fix ID");
    fix_event = (FixEvent*) modify->fix[ifix];
 
    if (strcmp(fix_event->style,"EVENT/PRD") != 0 &&
        strcmp(fix_event->style,"EVENT/TAD") != 0 &&
        strcmp(fix_event->style,"EVENT/HYPER") != 0)
      error->all(FLERR,"Compute event/dihedral has invalid fix event assigned");
  }

  // Why necessary? Recheck that group 2 has not been deleted

  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR,"Compute group/group group ID does not exist");
  jgroupbit = group->bitmask[jgroup];

}

/* ----------------------------------------------------------------------
   count dihedrals on this proc
   only count if 2nd atom is the one storing the dihedral
   all atoms in interaction must be in group
   all atoms in interaction must be known to proc
   if flag is set, compute requested info about dihedral
------------------------------------------------------------------------- */

double ComputeEventDihedral::compute_scalar()
{
  invoked_scalar = update->ntimestep;

  if (id_event == NULL) return 0.0;
  
  // default no event
  double event = 0.0;
  double **xevent = fix_event->array_atom;

  int i,nd,atom1,atom2,atom3,atom4,imol,iatom;
  tagint tagprev;

  // This is to keep track of if dihed. is in group 1, group 2, or both...
  // for now, should be either phi or psi 
  // hard-coding that group1 == phi and group2 == psi atom definitions
  int grpflag, jgrpflag;

  int dihedcount = 0;
  double dih, dihev, jdih, jdihev;
  int region, regionev;

  // I think don't need following line
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *num_dihedral = atom->num_dihedral;
  tagint **dihedral_atom1 = atom->dihedral_atom1;
  tagint **dihedral_atom2 = atom->dihedral_atom2;
  tagint **dihedral_atom3 = atom->dihedral_atom3;
  tagint **dihedral_atom4 = atom->dihedral_atom4;
  int *mask = atom->mask;

  int *molindex = atom->molindex;
  int *molatom = atom->molatom;
  Molecule **onemols = atom->avec->onemols;

  int nlocal = atom->nlocal;
  int molecular = atom->molecular;

/*   
  int numj;
  numj = group->count(jgroup);
  fprintf(universe->uscreen,"num atoms in group2: %d\n",numj);

  int testatom;
  for (testatom = 0; testatom < nlocal; testatom++) {
    if (mask[testatom] & groupbit) {
      fprintf(universe->uscreen,"atom %d in group2\n",testatom);
     }
  } 
*/

  for (atom2 = 0; atom2 < nlocal; atom2++) {
    grpflag = jgrpflag = 0;
    if (!(mask[atom2] & groupbit || mask[atom2] & jgroupbit)) continue;
    if (mask[atom2] & groupbit) grpflag = 1;
    if (mask[atom2] & jgroupbit) jgrpflag = 1;

    if (molecular == 1) nd = num_dihedral[atom2];
    else {
      if (molindex[atom2] < 0) continue;
      imol = molindex[atom2];
      iatom = molatom[atom2];
      nd = onemols[imol]->num_dihedral[iatom];
    }

    for (i = 0; i < nd; i++) {
      if (molecular == 1) {
        if (tag[atom2] != dihedral_atom2[atom2][i]) continue;
        atom1 = atom->map(dihedral_atom1[atom2][i]);
        atom3 = atom->map(dihedral_atom3[atom2][i]);
        atom4 = atom->map(dihedral_atom4[atom2][i]);
      } else {
        if (tag[atom2] != onemols[imol]->dihedral_atom2[atom2][i]) continue;
        tagprev = tag[atom2] - iatom - 1;
        atom1 = atom->map(onemols[imol]->dihedral_atom1[atom2][i]+tagprev);
        atom3 = atom->map(onemols[imol]->dihedral_atom3[atom2][i]+tagprev);
        atom4 = atom->map(onemols[imol]->dihedral_atom4[atom2][i]+tagprev);
      }

      if (!(atom1 < 0 || atom3 < 0 || atom4 < 0)) {
        if (grpflag) {
          if ((mask[atom1] & groupbit) && (mask[atom3] & groupbit) && (mask[atom4] & groupbit)) {
            dih = compute_dihedral(atom1, atom2, atom3, atom4, 0);
            dihev = compute_dihedral(atom1, atom2, atom3, atom4, 1);
//            fprintf(universe->uscreen,"atoms: %d, %d, %d, %d in grp1\n", atom1, atom2, atom3, atom4);
//            fprintf(universe->uscreen,"no fxn dih: %f\n", dih);
            dihedcount += 1;
          }
        }
        if (jgrpflag) {
          if ((mask[atom1] & jgroupbit) && (mask[atom3] & jgroupbit) && (mask[atom4] & jgroupbit)) {
            jdih = compute_dihedral(atom1, atom2, atom3, atom4, 0);
            jdihev = compute_dihedral(atom1, atom2, atom3, atom4, 1);
//            fprintf(universe->uscreen,"atoms: %d, %d, %d, %d in grp2\n", atom1, atom2, atom3, atom4);
            dihedcount += 1;
          }
        }
      }
      if (dihedcount == 2) break;
    }
  }

  region = find_region(dih, jdih);
  regionev = find_region(dihev, jdihev);

  if (region == 0 | regionev == 0) error->all(FLERR,"Region not defined");

  fprintf(universe->uscreen,"phi: %f, psi: %f\n", dih, jdih);
  fprintf(universe->uscreen,"evphi: %f, evpsi: %f\n", dihev, jdihev);
  fprintf(universe->uscreen,"prev region: %d, current: %d\n", regionev, region);


  if (region != regionev) {
    event = 1.0;
  }

  MPI_Allreduce(&event,&scalar,1,MPI_DOUBLE,MPI_SUM,world);

  return scalar;
}

/* ----------------------------------------------------------------------
   determine region of Ramachandran plot from two angles
------------------------------------------------------------------------- */
int ComputeEventDihedral::find_region(double phi, double psi)
{
  int region = 0;

  if ((0.0 <= phi) && (phi < 150.0)) {
    if ((0.0 <= psi) && (psi < 130.0)) region = 5;
    else region = 1;
  } else {
    if ((-120.0 <= psi) && (psi < 0.0)) region = 2;
    else if ((0.0 <= psi) && (psi < 120.0)) region = 4;
    else region = 3;
  }
  
  return region;
}

/* ----------------------------------------------------------------------
   actual dihedral computation from dihedral style harmonic
------------------------------------------------------------------------- */
double ComputeEventDihedral::compute_dihedral(int atom1, int atom2, int atom3, int atom4, int eventflag) 
{
  double vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y,vb3z,vb2xm,vb2ym,vb2zm;
  double ax,ay,az,bx,by,bz,rasq,rbsq,rgsq,rg,ra2inv,rb2inv,rabinv;
  double s,c, dih;
  double **x = NULL ;

  if (eventflag) {
    x = fix_event->array_atom ;
  } else {
    x = atom->x ;
  }

  vb1x = x[atom1][0] - x[atom2][0];
  vb1y = x[atom1][1] - x[atom2][1];
  vb1z = x[atom1][2] - x[atom2][2];
  domain->minimum_image(vb1x,vb1y,vb1z);

  vb2x = x[atom3][0] - x[atom2][0];
  vb2y = x[atom3][1] - x[atom2][1];
  vb2z = x[atom3][2] - x[atom2][2];
  domain->minimum_image(vb2x,vb2y,vb2z);

  vb2xm = -vb2x;
  vb2ym = -vb2y;
  vb2zm = -vb2z;
  domain->minimum_image(vb2xm,vb2ym,vb2zm);

  vb3x = x[atom4][0] - x[atom3][0];
  vb3y = x[atom4][1] - x[atom3][1];
  vb3z = x[atom4][2] - x[atom3][2];
  domain->minimum_image(vb3x,vb3y,vb3z);

  ax = vb1y*vb2zm - vb1z*vb2ym;
  ay = vb1z*vb2xm - vb1x*vb2zm;
  az = vb1x*vb2ym - vb1y*vb2xm;
  bx = vb3y*vb2zm - vb3z*vb2ym;
  by = vb3z*vb2xm - vb3x*vb2zm;
  bz = vb3x*vb2ym - vb3y*vb2xm;

  rasq = ax*ax + ay*ay + az*az;
  rbsq = bx*bx + by*by + bz*bz;
  rgsq = vb2xm*vb2xm + vb2ym*vb2ym + vb2zm*vb2zm;
  rg = sqrt(rgsq);

  ra2inv = rb2inv = 0.0;
  if (rasq > 0) ra2inv = 1.0/rasq;
  if (rbsq > 0) rb2inv = 1.0/rbsq;
  rabinv = sqrt(ra2inv*rb2inv);

  c = (ax*bx + ay*by + az*bz)*rabinv;
  s = rg*rabinv*(ax*vb3x + ay*vb3y + az*vb3z);

  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;
  dih = 180.0*atan2(s,c)/MY_PI;

//  fprintf(universe->uscreen,"in fxn dih: %f\n",dih);

  return dih;
}

/* ---------------------------------------------------------------------- */

void ComputeEventDihedral::reset_extra_compute_fix(const char *id_new)
{
  delete [] id_event;
  id_event = NULL;
  if (id_new == NULL) return;

  int n = strlen(id_new) + 1;
  id_event = new char[n];
  strcpy(id_event,id_new);
}

