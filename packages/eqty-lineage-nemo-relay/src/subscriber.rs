//! The event subscriber, registered through the raw ABI so that events are decoded by us.
//!
//! `ssi` (via `did-ethr`) turns on serde_json's `arbitrary_precision` for this whole cdylib. Under
//! that feature, `from_str` cannot decode an `f64` that sits behind `#[serde(flatten)]` -- the
//! buffered number arrives as a map -- and Relay's `CategoryProfile::time_to_first_chunk` is one.
//! The SDK's typed `register_subscriber` decodes with `from_str` and drops the event on error, so
//! every streaming LLM scope end, the half carrying the response, would never reach us. Decoding
//! through a `Value` first does not hit that path.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use nemo_relay_plugin::{
    Event, NemoRelayNativeHostApiV1, NemoRelayNativeString, NemoRelayStatus, PluginContext, Result,
};

/// Decode one event as Relay serialized it.
pub fn decode_event(json: &str) -> Option<Event> {
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    serde_json::from_value(value).ok()
}

struct Subscriber<F> {
    host: NemoRelayNativeHostApiV1,
    on_event: F,
    on_undecodable: Box<dyn Fn() + Send + Sync>,
}

/// Register `on_event` for every event Relay emits; `on_undecodable` counts the ones it could not.
pub fn register<F>(
    ctx: &mut PluginContext<'_>,
    name: &str,
    on_event: F,
    on_undecodable: Box<dyn Fn() + Send + Sync>,
) -> Result<()>
where
    F: Fn(&Event) + Send + Sync + 'static,
{
    let user_data = Box::into_raw(Box::new(Subscriber {
        host: *ctx.host_api(),
        on_event,
        on_undecodable,
    })) as *mut c_void;
    // The host owns `user_data` from here, and frees it through `release` on failure too.
    let status = unsafe {
        ctx.register_subscriber_raw(name, trampoline::<F>, user_data, Some(release::<F>))
    };
    match status {
        NemoRelayStatus::Ok => Ok(()),
        status => Err(format!("subscriber registration failed: {status:?}")),
    }
}

unsafe extern "C" fn trampoline<F>(
    user_data: *mut c_void,
    event_json: *const NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(&Event) + Send + Sync + 'static,
{
    if user_data.is_null() || event_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let state = unsafe { &*(user_data as *const Subscriber<F>) };
    // A panic must not unwind across `extern "C"`.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let text = unsafe { read(&state.host, event_json) };
        match text.as_deref().and_then(decode_event) {
            Some(event) => (state.on_event)(&event),
            None => (state.on_undecodable)(),
        }
    }));
    match outcome {
        Ok(()) => NemoRelayStatus::Ok,
        Err(_) => NemoRelayStatus::Internal,
    }
}

unsafe extern "C" fn release<F>(user_data: *mut c_void) {
    if !user_data.is_null() {
        drop(unsafe { Box::from_raw(user_data as *mut Subscriber<F>) });
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
