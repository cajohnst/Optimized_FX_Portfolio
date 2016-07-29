ln -s -f /usr/lib/x86_64-linux-gnu/mesa/libGL.so $PWD/.apt/usr/lib/x86_64-linux-gnu/libGL.so
ln -s -f /usr/lib/x86_64-linux-gnu/mesa/libGL.so.1 $PWD/.apt/usr/lib/x86_64-linux-gnu/libGL.so


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
