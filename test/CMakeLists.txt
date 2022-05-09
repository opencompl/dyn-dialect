configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(DYN_TEST_DEPENDS
        FileCheck count not
        dyn-opt
        )

add_lit_testsuite(check-dyn "Running the dyn regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${DYN_TEST_DEPENDS}
        )
set_target_properties(check-dyn PROPERTIES FOLDER "Tests")

add_lit_testsuites(DYN ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${DYN_TEST_DEPENDS})
