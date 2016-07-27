printenv
ls -l $PWD/.apt/usr/bin/
ln -s -f $PWD/.apt/usr/lib/x86_64-linux-gnu/qt5/bin/qmake $PWD/.apt/usr/bin/qmake
export QMAKESPEC=$PWD/.apt/usr/lib/x86_64-linux-gnu/qt5/mkspecs/linux-g++
ls -l $QMAKESPEC/..
printenv
echo $QMAKESPEC