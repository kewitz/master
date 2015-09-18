// Gmsh project created on Tue Sep 15 15:17:24 2015
DefineConstant[ in = { 0.1, Path "Gmsh/Parameters"}];
DefineConstant[ out = { 0.1, Path "Gmsh/Parameters"}];
Point(1) = {0, 0.0, 0, out};
Point(2) = {0, 1, 0, out};
Point(3) = {0.5, 1, 0, in};
Point(4) = {0.5, 0.5, 0, in};
Point(5) = {1, 0.5, 0, in};
Point(6) = {1, 0, 0, out};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line Loop(7) = {2, 3, 4, 5, 6, 1};
Plane Surface(8) = {7};

