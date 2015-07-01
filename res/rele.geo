// Gmsh project created on Mon Jun 29 14:39:10 2015
DefineConstant[ detail = { 0.01, Label "Detalhe", Path "Gmsh/Parameters"}];
DefineConstant[ dist = { 0.02, Label "Distância Peça", Path "Gmsh/Parameters"}];
// Bounds
Point(1) = {0, 0, 0, detail};
Point(2) = {0.12, 0, 0, detail};
Point(3) = {0.12, 0.1, 0, detail};
Point(4) = {0, 0.1, 0, detail};
// Ferrite
Point(5) = {0, 0.02, 0, detail};
Point(6) = {0, 0.08, 0, detail};
Point(7) = {0.06, 0.02, 0, detail};
Point(8) = {0.06, 0.08, 0, detail};
Point(9) = {0.06, 0.03, 0, detail};
Point(10) = {0.06, 0.07, 0, detail};
Point(11) = {0.02, 0.03, 0, detail};
Point(12) = {0.02, 0.07, 0, detail};
// Coil
Point(13) = {0.03, 0.03, 0, detail};
Point(14) = {0.03, 0.07, 0, detail};
// Piece
Point(15) = {0.06+dist, 0.02, 0, detail};
Point(16) = {0.06+dist, 0.08, 0, detail};
Point(17) = {0.06+dist+.01, 0.02, 0, detail};
Point(18) = {0.06+dist+.01, 0.08, 0, detail};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {4, 3};
Line(4) = {4, 6};
Line(5) = {6, 5};
Line(6) = {5, 1};
Line(7) = {5, 7};
Line(8) = {7, 9};
Line(9) = {9, 13};
Line(10) = {13, 11};
Line(11) = {11, 12};
Line(12) = {12, 14};
Line(13) = {14, 10};
Line(14) = {10, 8};
Line(15) = {8, 6};
Line(16) = {16, 15};
Line(17) = {17, 15};
Line(18) = {17, 18};
Line(19) = {18, 16};
Line(20) = {14, 13};
Line Loop(21) = {20, 10, 11, 12};
Plane Surface(22) = {21};
Line Loop(23) = {13, 14, 15, 5, 7, 8, 9, 10, 11, 12};
Plane Surface(24) = {23};
Line Loop(25) = {16, -17, 18, 19};
Plane Surface(26) = {25};
Line Loop(27) = {2, -3, 4, -15, -14, -13, 20, -9, -8, -7, 6, 1};
Plane Surface(28) = {25, 27};
