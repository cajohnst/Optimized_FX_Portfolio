printenv

ln -s -f $PWD/.apt/usr/lib/x86_64-linux-gnu/qt5/bin/qmake $PWD/.apt/usr/bin/qmake
echo "QMAKE LOCATION"
which qmake

qmake -query

PREFIX="$PWD/.apt/usr"
ARCHDATA="lib/x86_64-linux-gnu/qt5"
DATA="share/qt5"
DOCUMENTATION="share/qt5/doc"
HEADERS="include/qt5"
LIBRARIES="lib/x86_64-linux-gnu"
LIBRARYEXECUTABLES="lib/x86_64-linux-gnu/qt5/libexec"
BINARIES="lib/x86_64-linux-gnu/qt5/bin"
PLUGINS="lib/x86_64-linux-gnu/qt5/plugins"
IMPORTS="lib/x86_64-linux-gnu/qt5/imports"
TRANSLATIONS="share/qt5/translations"
CONFIGURATION="/etc/xdg"


echo "[Paths]
Prefix = $PREFIX
Documentation = $DOCUMENTATION
Headers = $HEADERS
Libraries = $LIBRARIES
LibraryExecutables = $LIBRARYEXECUTABLES
Binaries = $BINARIES
Plugins = $PLUGINS
Imports = $IMPORTS
Translations = $TRANSLATIONS
ArchData = $ARCHDATA
Data = $DATA" > $PWD/.apt/usr/bin/qt.conf


cat $PWD/.apt/usr/bin/qt.conf

echo "set"
qmake -query