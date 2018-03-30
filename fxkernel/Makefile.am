# Always set this so that we can just '#include "foo.hpp"' from anywhere,
# including our tests directory.
AM_CPPFLAGS = -I$(top_srcdir)/src

# This is the top-level Makefile.am for a non-recursive Autotools setup. In
# fact, in non-recursive structures, this is the only Makefile.am that is
# needed. The programs (just foo.exe in this case) will be left in the top-level
# directory.
bin_PROGRAMS = testfxkernel

foo_SOURCES = \
  src/testfxkernel.cpp 
