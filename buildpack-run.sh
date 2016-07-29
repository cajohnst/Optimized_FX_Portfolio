printenv

ln -s -f $PWD/.apt/usr/lib/x86_64-linux-gnu/qt5/bin/qmake $PWD/.apt/usr/bin/qmake
echo "QMAKE LOCATION"
which qmake

ls -l /usr/lib/x86_64-linux-gnu
echo "NEXTTT

"
ls -l /usr/lib/x86_64-linux-gnu/mesa

ln -s -f /usr/lib/x86_64-linux-gnu/mesa/libGL.so $PWD/.apt/usr/lib/x86_64-linux-gnu/libGL.so
ln -s -f /usr/lib/x86_64-linux-gnu/mesa/libGL.so.1 $PWD/.apt/usr/lib/x86_64-linux-gnu/libGL.so

qmake -query

PREFIX="$PWD/.apt/usr/lib/x86_64-linux-gnu/qt5"
DOCUMENTATION="../../../share/qt5/doc"
HEADERS="../../../include/qt5"
LIBRARIES=".."
TRANSLATIONS="../../../share/qt5/translations"


echo "[Paths]
Prefix = $PREFIX
Documentation = $DOCUMENTATION
Headers = $HEADERS
Libraries = $LIBRARIES
Translations = $TRANSLATIONS" > $PWD/.apt/usr/bin/qt.conf

cat $PWD/.apt/usr/bin/qt.conf

echo "set"
qmake -query