#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/* General coordinate rotations. All in radians.
 * Input and output systems:
 * 0 = Equatorial   J2000
 * 1 = Ecliptic     J2000
 * 2 = Galactic
 */
void rotate_coords(double lat_in, double lon_in, double *lat_out, double *lon_out, int sys_in, int sys_out) {
  
  double EqSys[] = {1,0,0, 0,1,0, 0,0,1};
  double EclSys[] = {1,0,0, 0,0.917482062840573,0.397777154152682, 0,-0.397777154152682,0.917482062840573};
  double GalSys[] = {-0.054876531,-0.873436658,-0.483835686, 0.494110662,-0.444830359,0.746980994, -0.867665385,-0.198076645,0.455985112};
  double *Si, *So;
  int i,j;
  double x_in[3], x_eq[3], x_out[3];
   
  /* Set up input & output coordinates */
  Si = EqSys;
  if (sys_in==1) Si = EclSys;
  if (sys_in==2) Si = GalSys;
                                                                                               
  So = EqSys;
  if (sys_out==1) So = EclSys;
  if (sys_out==2) So = GalSys;
    
  /* Now build the coordinate vector and do rotations */
  x_in[0] = cos(lat_in)*cos(lon_in);
  x_in[1] = cos(lat_in)*sin(lon_in);
  x_in[2] = sin(lat_in); 
    
  for(i=0;i<3;i++) {
    x_eq[i] = 0.;
    for(j=0;j<3;j++) x_eq[i] += Si[3*j+i]*x_in[j];
  }
   
  for(i=0;i<3;i++) {
    x_out[i] = 0.;
    for(j=0;j<3;j++) x_out[i] += So[3*i+j]*x_eq[j];
  }
  
  *lat_out = atan2(x_out[2], sqrt(x_out[0]*x_out[0]+x_out[1]*x_out[1]));
  *lon_out = atan2(-x_out[1], -x_out[0]) + M_PI;
}

/* Check whether an observation looks at a particular point.
 *
 * Inputs:
 *   obsRA, obsDec, obsPA = Euler angles of the observation
 *   lon, lat = longitude and latitude of the desired target
 *   coordsys = coordinate system of the *target* (0=J2000, 1=E2000, 2=Galactic)
 *
 * Important: all angles given are in **radians**, if your angle is in degrees you need to multiply by M_PI/180.
 *
 * Returns the chip number (1-18) if in the WFC field of view, 0 otherwise.
 * Fastest in input coordinate system 0 (equatorial) but works in general.
 */
int wfirst_get_chip_number(double obsRA, double obsDec, double obsPA,
  double lon, double lat, int coordsys) {

#define MAX_RAD_FROM_BORESIGHT 0.009

  unsigned long constraint_type = 0x3;
  int i;
  double xptsc[2], *cptr;
  double xi, yi, axi;
  double ptRA, ptDec; /* of target */
  double mX,mY,cpa,spa,cDec,sDec,cptDec,sptDec,cdRA;

  double *AFTA_SCA_Coords;
  static double AFTA_SCA_Coords_ImC[] = {
      0.002689724,  1.000000000,  0.181995021, -0.002070809, -1.000000000,  0.807383134,  1.000000000,  0.004769437,  1.028725015, -1.000000000, -0.000114163, -0.024579913,
      0.003307633,  1.000000000,  1.203503349, -0.002719257, -1.000000000, -0.230036847,  1.000000000,  0.006091805,  1.028993582, -1.000000000, -0.000145757, -0.024586416,
      0.003888409,  1.000000000,  2.205056241, -0.003335597, -1.000000000, -1.250685466,  1.000000000,  0.007389324,  1.030581048, -1.000000000, -0.000176732, -0.024624426,
      0.007871078,  1.000000000, -0.101157485, -0.005906926, -1.000000000,  1.095802866,  1.000000000,  0.009147586,  2.151242511, -1.000000000, -0.004917673, -1.151541644,
      0.009838715,  1.000000000,  0.926774753, -0.007965112, -1.000000000,  0.052835488,  1.000000000,  0.011913584,  2.150981875, -1.000000000, -0.006404157, -1.151413352,
      0.011694346,  1.000000000,  1.935534773, -0.009927853, -1.000000000, -0.974276664,  1.000000000,  0.014630945,  2.153506744, -1.000000000, -0.007864196, -1.152784334,
      0.011758070,  1.000000000, -0.527032681, -0.008410887, -1.000000000,  1.529873670,  1.000000000,  0.012002262,  3.264990040, -1.000000000, -0.008419930, -2.274065453,
      0.015128555,  1.000000000,  0.510881058, -0.011918799, -1.000000000,  0.478274989,  1.000000000,  0.016194244,  3.262719942, -1.000000000, -0.011359106, -2.272508364,
      0.018323436,  1.000000000,  1.530828790, -0.015281655, -1.000000000, -0.558879607,  1.000000000,  0.020320244,  3.264721809, -1.000000000, -0.014251259, -2.273955111,
     -0.002689724,  1.000000000,  0.181995021,  0.002070809, -1.000000000,  0.807383134,  1.000000000, -0.000114163, -0.024579913, -1.000000000,  0.004769437,  1.028725015,
     -0.003307633,  1.000000000,  1.203503349,  0.002719257, -1.000000000, -0.230036847,  1.000000000, -0.000145757, -0.024586416, -1.000000000,  0.006091805,  1.028993582,
     -0.003888409,  1.000000000,  2.205056241,  0.003335597, -1.000000000, -1.250685466,  1.000000000, -0.000176732, -0.024624426, -1.000000000,  0.007389324,  1.030581048,
     -0.007871078,  1.000000000, -0.101157485,  0.005906926, -1.000000000,  1.095802866,  1.000000000, -0.004917673, -1.151541644, -1.000000000,  0.009147586,  2.151242511,
     -0.009838715,  1.000000000,  0.926774753,  0.007965112, -1.000000000,  0.052835488,  1.000000000, -0.006404157, -1.151413352, -1.000000000,  0.011913584,  2.150981875,
     -0.011694346,  1.000000000,  1.935534773,  0.009927853, -1.000000000, -0.974276664,  1.000000000, -0.007864196, -1.152784334, -1.000000000,  0.014630945,  2.153506744,
     -0.011758070,  1.000000000, -0.527032681,  0.008410887, -1.000000000,  1.529873670,  1.000000000, -0.008419930, -2.274065453, -1.000000000,  0.012002262,  3.264990040,
     -0.015128555,  1.000000000,  0.510881058,  0.011918799, -1.000000000,  0.478274989,  1.000000000, -0.011359106, -2.272508364, -1.000000000,  0.016194244,  3.262719942,
     -0.018323436,  1.000000000,  1.530828790,  0.015281655, -1.000000000, -0.558879607,  1.000000000, -0.014251259, -2.273955111, -1.000000000,  0.020320244,  3.264721809
  };
  AFTA_SCA_Coords = AFTA_SCA_Coords_ImC;

  /* Get target position in equatorial coordinates */
  if (coordsys) {
    rotate_coords(lat,lon,&ptDec,&ptRA,coordsys,0);
  } else {
    ptDec=lat; ptRA=lon;
  }

  /* If the object is more than some encircling radius away from the boresight, give up
   * to save computing time.
   */
  if (fabs(obsDec-ptDec)>MAX_RAD_FROM_BORESIGHT) {
    return(0);
  }
  cDec = cos(obsDec); sDec = sin(obsDec);
  cptDec = cos(ptDec); sptDec = sin(ptDec);
  cdRA = cos(obsRA-ptRA);
  if (sptDec*sDec+cptDec*cDec*cdRA<cos(MAX_RAD_FROM_BORESIGHT))
    return(0);

  /* Position of the object in boresight coordinates */
  mX = -sDec*cptDec*cdRA + cDec*sptDec;
  mY = cptDec*sin(obsRA-ptRA);
  xptsc[0] = (cpa=cos(obsPA))*mX - (spa=sin(obsPA))*mY;
  xptsc[1] = spa*mX+cpa*mY;

  /* NOW the FCC coordinates of the test point are (xptsc[0], xptsc[1], ?)
   * [The FCC Z-coordinate is positive if we get here and not independently needed.]
   */

  switch(constraint_type) {

    case 0x2:
    case 0x3:
      /* 6x3 distorted */
      xi = -xptsc[1]/0.0021801102; /* Image plane position in chips */
      axi = fabs(xi);
      if (axi>3.4) return(0);
      yi =  xptsc[0]/0.0021801102;
      if (fabs(yi)>2.6) return(0);
      for(i=0;i<18;i++) {
        cptr = AFTA_SCA_Coords + 12*i;
        if (cptr[0]*xi+cptr[1]*yi<cptr[2] && cptr[3]*xi+cptr[4]*yi<cptr[5]
          && cptr[6]*xi+cptr[7]*yi<cptr[8] && cptr[9]*xi+cptr[10]*yi<cptr[11]) {
          return(i+1);
        }
      }
      return(0);
      break;

    default:
      fprintf(stderr, "Error: illegal constraint type.\n");
      exit(1);
      break;
  }
  return(1);

#undef MAX_RAD_FROM_BORESIGHT
}

int main()
{
  /*
  double obsRA = 0.44514099;
  double obsDec = -0.47871763;
  double obsPA = 3.7089991934131494;
  double lon = -0.475768134717;
  double lat = 0.452246096019;
  int coordsys = 0;
  int chip;
  chip = wfirst_get_chip_number(obsRA, obsDec, obsPA, lon, lat, coordsys);
  printf("%d\n", chip);
  return 0;
  */
  FILE *myFile,*fobsRA,*fobsDec,*fobsPA,*flen;
  myFile  = fopen("coords.txt", "r");
  fobsRA  = fopen("obsra.txt","r");
  fobsDec = fopen("obsdec.txt","r");
  fobsPA  = fopen("obspa.txt","r");
  flen    = fopen("len.txt","r");

  double radcon = 0.017453292519943295;
  //double obsRA=0.44514099;
  //double obsDec=-0.47871763;
  double obsRA=0.4505957241299;
  double obsDec=-1.483529864195;
  double obsPA=0.069666224;
  int i, chip, len;
  int coordsys = 0;

  fscanf(fobsRA,  "%lf", &obsRA);
  fscanf(fobsDec, "%lf", &obsDec);
  fscanf(fobsPA,  "%lf", &obsPA);

  fscanf(flen,    "%ld", &len);
  double lon[len];
  double lat[len];

  for(i=0;i<len;i++) {
    fscanf(myFile, "%lf %lf", &lat[i], &lon[i]);

    chip = wfirst_get_chip_number(obsRA, obsDec, obsPA, lat[i]*radcon, lon[i]*radcon, coordsys);
    printf("%d\n", chip);
    /*
    printf("%e %e\n", lon[i]*radcon, lat[i]*radcon);
    */
  }
  return 0;
}
