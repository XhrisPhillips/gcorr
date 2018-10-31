#AC_INIT([FULL-PACKAGE-NAME], [VERSION], [BUG-REPORT-ADDRESS], [TAR-NAME], [URL])
AC_INIT([fxkernel], [0.0.1], [adeller@gmail.com], [fxkernel], [https://github.com/XhrisPhillips/gcorr])

# Force autoconf to be at least this version number.
AC_PREREQ([2.68])

# Put autotools auxiliary files in a subdir, so they don't clutter top dir.
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])

# Enable "automake" to simplify creating makefiles. foreign relaxes some GNU
# checks. -Wall and -Werror are instructions to Automake, not gcc.
# Because this is a non-recursive automake, we use subdir-objects to tell
# automake to create objects such as .o in sub-directories rather than placing
# all artifacts in the top level directory.
AM_INIT_AUTOMAKE([subdir-objects foreign -Wall -Werror])

# Need to have IPP
PKG_CHECK_MODULES(IPP, ipp)
AC_SUBST(IPP_CFLAGS)
AC_SUBST(IPP_LIBS)

# Use C++11
AX_CXX_COMPILE_STDCXX([11], [noext], [mandatory])

# These are the files to be generated.
AC_CONFIG_FILES([Makefile src/Makefile])

# Safety check - list a source file that wouldn't be in other directories.
AC_CONFIG_SRCDIR([src/testfxkernel.cpp])

# Put configuration results in this file for us to #include in our program.
AC_CONFIG_HEADERS([config.h])

# Check for a C compiler.
AC_PROG_CC

# Check for a C++ compiler.
AC_PROG_CXX

# Check for libraries, headers, data etc here.
CXXFLAGS="${CXXFLAGS} ${IPP_CFLAGS}"
LIBS="${IPP_LIBS} $LIBS"

# Last line: actually generate all required output files.
AC_OUTPUT
