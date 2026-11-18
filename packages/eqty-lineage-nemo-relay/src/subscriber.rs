//! The event subscriber, registered through the raw ABI so that events are decoded by us.
//!
//! `ssi` turns on serde_json's `arbitrary_precision` for this whole cdylib, and under it `from_str`
//! cannot decode an `f64` behind `#[serde(flatten)]`, which every streamed LLM scope end carries.
//! The SDK's typed `register_subscriber` decodes that way and drops the event on error.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use nemo_relay_plugin::{
    Event, Json, NemoRelayNativeHostApiV1, NemoRelayNativeString, NemoRelayStatus, PluginContext,
    Result,
};

/// What the handler is given for one host callback.
pub enum Delivery<'a> {
    Event(&'a Event),
    /// The event's JSON when it parsed but did not decode, and `None` when it did not parse.
    Undecodable(Option<&'a Json>),
    /// Handling the event panicked, and the panic was contained.
    Panicked,
}

pub type Handler = Box<dyn Fn(Delivery<'_>) + Send + Sync>;

/// Decode one event as Relay serialized it.
pub fn decode_event(json: &str) -> Option<Event> {
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    serde_json::from_value(value).ok()
}

struct Subscriber {
    host: NemoRelayNativeHostApiV1,
    handler: Handler,
}

/// Register `handler` for every event Relay emits.
pub fn register(ctx: &mut PluginContext<'_>, name: &str, handler: Handler) -> Result<()> {
    let user_data = Box::into_raw(Box::new(Subscriber {
        host: *ctx.host_api(),
        handler,
    })) as *mut c_void;
    // The host owns `user_data` from here, and frees it through `release` on failure too.
    let status = unsafe { ctx.register_subscriber_raw(name, trampoline, user_data, Some(release)) };
    match status {
        NemoRelayStatus::Ok => Ok(()),
        status => Err(format!("subscriber registration failed: {status:?}")),
    }
}

unsafe extern "C" fn trampoline(
    user_data: *mut c_void,
    event_json: *const NemoRelayNativeString,
) -> NemoRelayStatus {
    if user_data.is_null() || event_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let state = unsafe { &*(user_data as *const Subscriber) };
    // A panic must not unwind across `extern "C"`.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let text = unsafe { read(&state.host, event_json) };
        match text.as_deref().and_then(decode_event) {
            Some(event) => {
                (state.handler)(Delivery::Event(&event));
                true
            }
            None => {
                let raw = text.and_then(|text| serde_json::from_str::<Json>(&text).ok());
                (state.handler)(Delivery::Undecodable(raw.as_ref()));
                false
            }
        }
    }));
    // A non-`Ok` status puts the loss in the host's diagnostics. The host sets the last error for
    // it, replacing any set here.
    match outcome {
        Ok(true) => NemoRelayStatus::Ok,
        Ok(false) => NemoRelayStatus::InvalidJson,
        Err(_) => {
            let _ = catch_unwind(AssertUnwindSafe(|| (state.handler)(Delivery::Panicked)));
            NemoRelayStatus::Internal
        }
    }
}

unsafe extern "C" fn release(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    let state = unsafe { Box::from_raw(user_data as *mut Subscriber) };
    let host = state.host;
    // Dropping the last mailbox handle flushes sessions and joins the worker; a panic in that must
    // not unwind across `extern "C"`.
    if catch_unwind(AssertUnwindSafe(|| drop(state))).is_err() {
        set_last_error(&host, "subscriber state drop panicked");
    }
}

fn set_last_error(host: &NemoRelayNativeHostApiV1, message: &str) {
    let mut handle = std::ptr::null_mut();
    let status = unsafe { (host.string_new)(message.as_ptr(), message.len(), &mut handle) };
    if status == NemoRelayStatus::Ok && !handle.is_null() {
        unsafe {
            (host.last_error_set)(handle);
            (host.string_free)(handle);
        }
    }
}

unsafe fn read(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> Option<String> {
    let len = unsafe { (host.string_len)(value) };
    if len == 0 {
        return Some(String::new());
    }
    let data = unsafe { (host.string_data)(value) };
    if data.is_null() {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    std::str::from_utf8(bytes).ok().map(str::to_owned)
}
