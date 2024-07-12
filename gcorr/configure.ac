#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.71])
AC_INIT([gcorr],[0.0.1],[adeller@gmail.com],[gcorr],[https://github.com/XhrisPhillips/gcorr])
#AM_INIT_AUTOMAKE([pippo], [2.6])
AC_CONFIG_SRCDIR([src/gxkernel.h])
AC_CONFIG_HEADERS([config.h])

AM_INIT_AUTOMAKE([subdir-objects])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

AC_CHECK_PROG(NVCC_CHECK,nvcc,yes)
AS_IF([test x"$NVCC_CHECK" != x"yes"], [AC_MSG_ERROR([Please install nvcc before configuring.])])

# Checks for libraries.
# FIXME: Replace `main' with a function in `-lcudart':
#AC_CHECK_LIB([cudart], [main])
# FIXME: Replace `main' with a function in `-lcufft':
#AC_CHECK_LIB([cufft], [main])
# FIXME: Replace `main' with a function in `-lcurand':
#AC_CHECK_LIB([curand], [main])
# FIXME: Replace `main' with a function in `-lineinfo':
#AC_CHECK_LIB([ineinfo], [main])
# FIXME: Replace `main' with a function in `-lm':
#AC_CHECK_LIB([m], [main])

# Checks for header files.
AC_CHECK_HEADERS([arpa/inet.h fcntl.h float.h limits.h netdb.h stddef.h stdlib.h string.h strings.h sys/socket.h sys/time.h sys/timeb.h unistd.h wchar.h wctype.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_FUNC_FORK
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_FUNC_STRERROR_R
AC_CHECK_FUNCS([dup2 fchdir getcwd getpagesize gettimeofday localtime_r memset mkdir munmap regcomp rmdir socket strcasecmp strchr strdup strerror strrchr strstr strtol strtoull])

#AC_CONFIG_FILES([Makefile
             #    src/Makefile])
AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
