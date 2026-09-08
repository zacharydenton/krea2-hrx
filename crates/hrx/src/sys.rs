//! Raw declarations for the libhrx C API, mirroring `hrx_runtime.h`.
//!
//! Only what this runtime dispatches through is declared. Every handle is an
//! opaque pointer; a status is a pointer that is null on success.
#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

pub type hrx_status_t = *mut c_void;
pub type hrx_device_t = *mut c_void;
pub type hrx_stream_t = *mut c_void;
pub type hrx_buffer_t = *mut c_void;
pub type hrx_executable_t = *mut c_void;

/// `hrx_status_code_t`; only the value this runtime branches on is named.
pub const HRX_STATUS_ALREADY_EXISTS: c_int = 6;

/// `hrx_device_property_t::HRX_DEVICE_PROPERTY_ARCHITECTURE`.
pub const HRX_DEVICE_PROPERTY_ARCHITECTURE: c_int = 1;

pub const HRX_MEMORY_TYPE_HOST_VISIBLE: u32 = 0x0000_0002;
pub const HRX_MEMORY_TYPE_DEVICE_LOCAL: u32 = 0x0000_0030;
pub const HRX_BUFFER_USAGE_DEFAULT: u32 = 0x0000_0C03;
pub const HRX_BUFFER_USAGE_MAPPING_SCOPED: u32 = 0x0100_0000;
pub const HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct hrx_dispatch_config_t {
    pub workgroup_count: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub subgroup_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct hrx_buffer_ref_t {
    pub buffer: hrx_buffer_t,
    pub offset: usize,
    pub length: usize,
}

extern "C" {
    pub fn hrx_status_code(status: hrx_status_t) -> c_int;
    pub fn hrx_status_to_string(
        status: hrx_status_t,
        message: *mut *mut c_char,
        size: *mut usize,
    ) -> hrx_status_t;
    pub fn hrx_status_free_message(message: *mut c_char);
    pub fn hrx_status_ignore(status: hrx_status_t);

    pub fn hrx_gpu_initialize(flags: u32) -> hrx_status_t;
    pub fn hrx_gpu_shutdown() -> hrx_status_t;
    pub fn hrx_gpu_device_count(count: *mut c_int) -> hrx_status_t;
    pub fn hrx_gpu_device_get(index: c_int, device: *mut hrx_device_t) -> hrx_status_t;

    pub fn hrx_device_get_property(
        device: hrx_device_t,
        property: c_int,
        value: *mut c_void,
        value_size: usize,
    ) -> hrx_status_t;
    pub fn hrx_device_release(device: hrx_device_t);

    pub fn hrx_stream_create(
        device: hrx_device_t,
        flags: u32,
        stream: *mut hrx_stream_t,
    ) -> hrx_status_t;
    pub fn hrx_stream_release(stream: hrx_stream_t);
    pub fn hrx_stream_synchronize(stream: hrx_stream_t) -> hrx_status_t;
    pub fn hrx_stream_execution_barrier(stream: hrx_stream_t) -> hrx_status_t;
    pub fn hrx_stream_fill_buffer(
        stream: hrx_stream_t,
        buffer: hrx_buffer_t,
        offset: usize,
        size: usize,
        pattern: *const c_void,
        pattern_size: usize,
    ) -> hrx_status_t;
    pub fn hrx_stream_copy_buffer(
        stream: hrx_stream_t,
        src: hrx_buffer_t,
        src_offset: usize,
        dst: hrx_buffer_t,
        dst_offset: usize,
        size: usize,
    ) -> hrx_status_t;
    pub fn hrx_stream_dispatch(
        stream: hrx_stream_t,
        executable: hrx_executable_t,
        export_ordinal: u32,
        config: *const hrx_dispatch_config_t,
        constants: *const c_void,
        constants_size: usize,
        bindings: *const hrx_buffer_ref_t,
        binding_count: usize,
        flags: u32,
    ) -> hrx_status_t;

    pub fn hrx_buffer_allocate(
        stream: hrx_stream_t,
        size: usize,
        memory_type: u32,
        usage: u32,
        buffer: *mut hrx_buffer_t,
    ) -> hrx_status_t;
    pub fn hrx_buffer_release(buffer: hrx_buffer_t);
    pub fn hrx_buffer_get_device_ptr(
        buffer: hrx_buffer_t,
        device_ptr: *mut *mut c_void,
    ) -> hrx_status_t;

    pub fn hrx_synchronous_h2d(
        device: hrx_device_t,
        host_src: *const c_void,
        dst: hrx_buffer_t,
        dst_offset: usize,
        size: usize,
    ) -> hrx_status_t;
    pub fn hrx_synchronous_d2h(
        device: hrx_device_t,
        src: hrx_buffer_t,
        src_offset: usize,
        host_dst: *mut c_void,
        size: usize,
    ) -> hrx_status_t;

    pub fn hrx_executable_load_file(
        device: hrx_device_t,
        path: *const c_char,
        target_family: *const c_char,
        target_key: *const c_char,
        executable: *mut hrx_executable_t,
    ) -> hrx_status_t;
    pub fn hrx_executable_lookup_export_by_name(
        executable: hrx_executable_t,
        name: *const c_char,
        export_ordinal: *mut u32,
    ) -> hrx_status_t;
    pub fn hrx_executable_release(executable: hrx_executable_t);
}

/// `hrx_status_is_ok` is a static inline in the header, so it is reproduced here.
pub fn is_ok(status: hrx_status_t) -> bool {
    status.is_null()
}
