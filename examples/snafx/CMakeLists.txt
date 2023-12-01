CPMAddPackage(
    NAME libfmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 9.1.0
)
CPMAddPackage(
    NAME libsndfile
    GIT_REPOSITORY https://github.com/libsndfile/libsndfile
    GIT_TAG v1.0.30
    OPTIONS
        "BUILD_PROGRAMS OFF"
        "BUILD_EXAMPLES OFF"
        "BUILD_TESTING OFF"
)

create_example(snafx)
target_link_libraries(snafx
    PRIVATE
        fmt::fmt
        sndfile
)
