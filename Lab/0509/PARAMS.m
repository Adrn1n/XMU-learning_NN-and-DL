DIMS=[169,30,20,15,2,10];
alpha=0.1;beta=0.9;
% alpha=1e-3;beta=0.1;
ratio=0.1;rng(3);epoch=1:1000;
EPS=1e-6;
TRAIN_F_NAME="simpleDigits13by13_move.mat";TRAIN_LABEL_NAME="simpleDigits13by13_move_labels";TEST_F_NAME="simpleDigits13by13_origin.mat";TEST_LABEL_NAME="simpleDigits13by13_origin_labels";
DRAW_SIZE_TRAIN=50;DRAW_SIZE_TRAIN_MEAN=200;DRAW_SIZE_TEST=100;MARKER_TRAIN='o';MARKER_TRAIN_MEAN='x';MARKER_TEST='s';
NAMES=["1","2","3","4","5","6","7","8","9","10"];COLORS=lines(length(NAMES));
