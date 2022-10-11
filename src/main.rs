mod postwo;
mod smooth;

use std::{
    cell::RefCell,
    io::{stdin, BufRead},
    ops::{Add, Mul},
    sync::{Arc, Mutex},
    time::Instant,
};

use minifb::Scale;
use palette::float::Float;
use postwo::Pos2;
use rayon::prelude::*;

use audio_visualizer::dynamic::live_input::{list_input_devs, AudioDevAndCfg};
use audio_visualizer::dynamic::{live_input::setup_audio_input_loop, window_top_btm::TransformFn};
use cpal::{
    traits::{DeviceTrait, StreamTrait},
    StreamConfig,
};
use ringbuffer::{AllocRingBuffer, RingBuffer, RingBufferExt};

use smooth::Smooth;
use spectrum_analyzer::scaling::divide_by_N;
use spectrum_analyzer::windows::hann_window;
use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit, FrequencyValue};
use std::cmp::max;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
struct Settings {
    pub RESOLUTION: (usize, usize),
    pub ASPECTRATIO: f32,
    pub SCALE: Scale,
    pub TOTALSPEED: f32,
    pub MAXSPEED: f32,
    pub MINSPEED: f32,
    pub TOTALDAMPING: f32,
    pub MAXDAMPING: f32,
    pub MINDAMPING: f32,
    pub POINTCOUNT: usize,
    pub AVERAGECOLOR: f64,
    pub SCALINGFACTOR: f32,
    pub INVERTING: bool,
    pub SHOWCURSOR: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
struct SettingsFromFile {
    pub RESOLUTION: [usize; 2],
    pub SCALE: u8,
    pub TOTALSPEED: f32,
    pub MAXSPEED: f32,
    pub TOTALDAMPING: f32,
    pub MAXDAMPING: f32,
    pub POINTCOUNT: usize,
    pub AVERAGECOLOR: f64,
    pub SCALINGFACTOR: f32,
    pub INVERTING: bool,
    pub SHOWCURSOR: bool,
}

impl Settings {
    pub fn from_file(path: &str) -> Settings {
        let file = std::fs::read_to_string(path).unwrap();
        let settings_from_file: SettingsFromFile = serde_json::from_str(&file).unwrap();

        Settings {
            RESOLUTION: (
                settings_from_file.RESOLUTION[0],
                settings_from_file.RESOLUTION[1],
            ),
            ASPECTRATIO: settings_from_file.RESOLUTION[0] as f32
                / settings_from_file.RESOLUTION[1] as f32,
            SCALE: match settings_from_file.SCALE {
                1 => Scale::X1,
                2 => Scale::X2,
                4 => Scale::X4,
                8 => Scale::X8,
                16 => Scale::X16,
                32 => Scale::X32,
                _ => Scale::X1,
            },

            TOTALSPEED: settings_from_file.TOTALSPEED,

            MAXSPEED: settings_from_file.MAXSPEED * settings_from_file.TOTALSPEED,

            MINSPEED: settings_from_file.TOTALSPEED
                - (settings_from_file.MAXSPEED * settings_from_file.TOTALSPEED),

            TOTALDAMPING: settings_from_file.TOTALDAMPING,

            MAXDAMPING: settings_from_file.MAXDAMPING * settings_from_file.TOTALDAMPING,

            MINDAMPING: settings_from_file.TOTALDAMPING
                - (settings_from_file.MAXDAMPING * settings_from_file.TOTALDAMPING),

            POINTCOUNT: settings_from_file.POINTCOUNT,

            AVERAGECOLOR: settings_from_file.AVERAGECOLOR,

            SCALINGFACTOR: settings_from_file.SCALINGFACTOR,

            INVERTING: settings_from_file.INVERTING,

            SHOWCURSOR: settings_from_file.SHOWCURSOR,
        }
    }
}

impl Settings {
    pub fn default() -> Self {
        Self {
            RESOLUTION: (3840, 2160),
            ASPECTRATIO: 3840.0 / 2160.0,
            SCALE: Scale::X1,
            TOTALSPEED: 0.05,
            MAXSPEED: 0.05 * 0.99,
            MINSPEED: 0.05 - 0.05 * 0.99,
            TOTALDAMPING: 0.995,
            MAXDAMPING: 0.995 * 0.99,
            MINDAMPING: 0.995 - 0.995 * 0.99,
            POINTCOUNT: 100000,
            AVERAGECOLOR: 328.5,
            SCALINGFACTOR: 2.5,
            INVERTING: true,
            SHOWCURSOR: false,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Color {
    fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
    fn from_argb(argb: u32) -> Self {
        Self {
            r: ((argb >> 16) & 0xFF) as u8,
            g: ((argb >> 8) & 0xFF) as u8,
            b: (argb & 0xFF) as u8,
            a: ((argb >> 24) & 0xFF) as u8,
        }
    }
    fn from_hsva(h: f32, s: f32, v: f32, a: f32) -> Self {
        let h = h.fract();
        let s = s.max(0.0).min(1.0);
        let v = v.max(0.0).min(1.0);
        let a = a.max(0.0).min(1.0);

        let c = v * s;
        let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
        let m = v - c;
        let (r, g, b) = match (h * 6.0).floor() as u8 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            5 => (c, 0.0, x),
            _ => unreachable!(),
        };
        Self {
            r: ((r + m) * 255.0) as u8,
            g: ((g + m) * 255.0) as u8,
            b: ((b + m) * 255.0) as u8,
            a: (a * 255.0) as u8,
        }
    }
    fn to_argb(self) -> u32 {
        (self.a as u32) << 24 | (self.r as u32) << 16 | (self.g as u32) << 8 | (self.b as u32)
    }
}

impl Add<Color> for Color {
    type Output = Color;

    fn add(self, other: Color) -> Color {
        Color {
            r: (self.r as u64 + other.r as u64).min(255) as u8,
            g: (self.g as u64 + other.g as u64).min(255) as u8,
            b: (self.b as u64 + other.b as u64).min(255) as u8,
            a: (self.a as u64 + other.a as u64).min(255) as u8,
        }
    }
}

impl Mul<Color> for Color {
    type Output = Color;

    fn mul(self, other: Color) -> Color {
        Color {
            r: (self.r as u64 * other.r as u64 / 255).min(255) as u8,
            g: (self.g as u64 * other.g as u64 / 255).min(255) as u8,
            b: (self.b as u64 * other.b as u64 / 255).min(255) as u8,
            a: (self.a as u64 * other.a as u64 / 255).min(255) as u8,
        }
    }
}

fn main() {
    let mut settings = Settings::from_file("settings.json");

    let WIDTH: usize = match settings.SCALE {
        Scale::FitScreen => settings.RESOLUTION.0,
        Scale::X1 => settings.RESOLUTION.0,
        Scale::X2 => settings.RESOLUTION.0 / 2,
        Scale::X4 => settings.RESOLUTION.0 / 4,
        Scale::X8 => settings.RESOLUTION.0 / 8,
        Scale::X16 => settings.RESOLUTION.0 / 16,
        Scale::X32 => settings.RESOLUTION.0 / 32,
    };

    let HEIGHT: usize = match settings.SCALE {
        Scale::FitScreen => settings.RESOLUTION.1,
        Scale::X1 => settings.RESOLUTION.1,
        Scale::X2 => settings.RESOLUTION.1 / 2,
        Scale::X4 => settings.RESOLUTION.1 / 4,
        Scale::X8 => settings.RESOLUTION.1 / 8,
        Scale::X16 => settings.RESOLUTION.1 / 16,
        Scale::X32 => settings.RESOLUTION.1 / 32,
    };
    let visualize_spectrum: RefCell<Vec<(f64, f64)>> = RefCell::new(vec![(0.0, 0.0); 1024]);
    let to_spectrum_fn = move |audio: &[f32], sampling_rate: f32| {
        let skip_elements = audio.len() - 2048;

        let relevant_samples = &audio[skip_elements..skip_elements + 2048];

        let hann_window = hann_window(relevant_samples);
        let latest_spectrum = samples_fft_to_spectrum(
            &hann_window,
            sampling_rate as u32,
            FrequencyLimit::All,
            Some(&divide_by_N),
        )
        .unwrap();

        latest_spectrum
            .data()
            .iter()
            .zip(visualize_spectrum.borrow_mut().iter_mut())
            .for_each(|((fr_new, fr_val_new), (fr_old, fr_val_old))| {
                *fr_old = fr_new.val() as f64;
                let old_val = *fr_val_old * 0.84;
                let max = max(
                    *fr_val_new * 5000.0_f32.into(),
                    FrequencyValue::from(old_val as f32),
                );
                *fr_val_old = max.val() as f64;
            });

        visualize_spectrum.borrow().clone()
    };
    let in_dev = select_input_dev();
    let input_dev_and_cfg = AudioDevAndCfg::new(
        Some(in_dev),
        Some(StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(44100),
            buffer_size: cpal::BufferSize::Default,
        }),
    );
    let _audio_data_transform_fn = TransformFn::Complex(&to_spectrum_fn);

    let mut points = get_points(settings);

    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut window = minifb::Window::new(
        "Hello World - ESC to exit",
        WIDTH,
        HEIGHT,
        minifb::WindowOptions {
            resize: false,
            scale: settings.SCALE,
            borderless: true,
            transparency: true,
            none: true,
            topmost: true,
            ..minifb::WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut debounce = false;

    let sample_rate = input_dev_and_cfg.cfg().sample_rate.0 as f32;
    let latest_audio_data = init_ringbuffer(sample_rate as usize);
    let _audio_buffer_len = latest_audio_data.lock().unwrap().len();
    let stream = setup_audio_input_loop(latest_audio_data.clone(), input_dev_and_cfg);

    let _time_per_sample = 1.0 / sample_rate as f64;

    window.limit_update_rate(None);

    stream.play().unwrap();
    let mut current_target = Pos2 { x: 0., y: 0. };
    let _alternate = false;
    let mut deltatime = Instant::now();
    let mut spectrum = Vec::new();
    let mut max = Smooth::new(0.1);
    let mut hoffset = 0.;
    let mut scaling = true;
    let mut settingstest = false;
    max.speed = 0.1;
    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        let slice: Option<usize> = None;
        if let Ok(latest_audio_data) = latest_audio_data.clone().try_lock() {
            spectrum = latest_audio_data.to_vec();
            if let Some(slice) = slice {
                let skip_elements = spectrum.len() - slice;
                spectrum = spectrum[skip_elements..skip_elements + slice].to_vec();
            }
        }
        let _thismaxvalue = spectrum.iter().fold(0.0, |acc, x| -> f32 { acc.max(*x) });

        let maxindex = spectrum.len();

        let mut pointstodraw: Vec<Option<(usize, Color)>> = spectrum
            .par_iter()
            .enumerate()
            .map(|(i, f)| {
                let theta = ((i as f32 / maxindex as f32) + 0.25) * 2.0 * std::f32::consts::PI;
                let x = theta.cos() * f.abs().sqrt().sqrt().cos() / max.value;
                let y = theta.sin() * f.abs().sqrt().sqrt().cos() / max.value;

                let x = x + WIDTH as f32 / 2.0;
                let y = y + HEIGHT as f32 / 2.0;

                if x > 0.0 && x < WIDTH as f32 && y > 0.0 && y < HEIGHT as f32 {
                    let index = (x as usize) + (y as usize) * WIDTH;

                    Some((
                        index,
                        Color::from_hsva(
                            ((i as f64 / maxindex as f64) as f32 + hoffset) % 1.,
                            1.0,
                            1.0,
                            1.0,
                        ),
                    ))
                } else {
                    None
                }
            })
            .collect();

        if scaling {
            let nones = pointstodraw.par_iter().filter(|x| x.is_none()).count();
            if nones > 0 {
                max.target *= 1.5;
                scaling = false;
            } else {
                max.target = max.target * 1. - 0.001;
            }
        }

        current_target = Pos2 {
            x: spectrum[0] as f32 / 25.,
            y: spectrum[spectrum.len() - 1] as f32 / 25.,
        };

        buffer
            .par_iter_mut()
            .for_each(|x| *x = (Color::from_argb(*x) * Color::new(0, 0, 0, 255)).to_argb());

        let center = Pos2 {
            x: WIDTH as f32 / 2.0,
            y: HEIGHT as f32 / 2.0,
        };
        let mut stufftodraw = points
            .par_iter()
            .map(|(point, _color)| {
                let pos = (point.value * WIDTH as f32) + center;
                if pos.x >= 0. && pos.y >= 0. {
                    let x = pos.x as usize;
                    let y = pos.y as usize;
                    if (x < WIDTH) && (y < HEIGHT) {
                        Some((
                            x + y * WIDTH,
                            Color::from_hsva(
                                ((point.velocity.x.abs() + point.velocity.y.abs()) * 1000.).sqrt()
                                    % 1.,
                                1.0,
                                1.0,
                                1.0,
                            ),
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<Option<(usize, Color)>>>();
        if settings.SHOWCURSOR {
            let point = (current_target * WIDTH as f32) + center;

            let mut offsets: Vec<((f32, f32), Color)> = Vec::new();
            for x in -25..=25 {
                for y in -25..=25 {
                    let tx = x as f32;
                    let ty = y as f32;

                    let distfromorigin = (tx * tx + ty * ty).sqrt();
                    if distfromorigin <= 3. {
                        offsets
                            .push(((tx, ty), Color::from_hsva(distfromorigin / 20., 1., 1., 1.)));
                    }
                }
            }

            for (offset, color) in offsets {
                let thispoint = Pos2 {
                    x: point.x + offset.0,
                    y: point.y + offset.1,
                };
                if thispoint.x >= 0. && thispoint.y >= 0. {
                    let x = thispoint.x as usize;
                    let y = thispoint.y as usize;
                    if (x < WIDTH) && (y < HEIGHT) {
                        stufftodraw.push(Some((x + y * WIDTH, color)));
                    }
                }
            }
        }
        stufftodraw.append(&mut pointstodraw);
        for (index, color) in stufftodraw.into_iter().flatten() {
            buffer[index] = (Color::from_argb(buffer[index]) + color).to_argb();
        }

        let mouse_pos = window
            .get_mouse_pos(minifb::MouseMode::Clamp)
            .unwrap_or((0., 0.));
        let mouse_pos = Pos2 {
            x: (mouse_pos.0 / WIDTH as f32) - 0.5,
            y: ((mouse_pos.1 / HEIGHT as f32) - 0.5) / settings.ASPECTRATIO as f32,
        };

        let target = match window.get_mouse_down(minifb::MouseButton::Left) {
            true => mouse_pos,
            false => current_target,
        };
        let dt = deltatime.elapsed().as_secs_f32() * 100.;
        deltatime = Instant::now();
        points.par_iter_mut().for_each(|(smooth, _)| {
            smooth.update(dt);
            smooth.target = target;
        });
        max.update(dt);
        hoffset = (hoffset - 0.001) % 1. + 1.;
        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .unwrap_or_else(|e| {
                panic!("{}", e);
            });
        if window.get_mouse_down(minifb::MouseButton::Right) {
            if !debounce {
                debounce = true;
                points = get_points(settings);
            }
        } else {
            debounce = false;
        }
    }
}

fn get_points(settings: Settings) -> Vec<(Smooth<Pos2>, Color)> {
    let mut points: Vec<(Smooth<Pos2>, Color)> = Vec::new();
    for _ in 0..settings.POINTCOUNT {
        let randspeed = rand::random::<f32>();
        let randdamp = rand::random::<f32>();

        let color = Color::from_hsva(randspeed + randdamp / 2., 1.0, 1.0, 1.0);
        let speed = (randspeed * settings.MAXSPEED) + settings.MINSPEED;
        let damping = (randdamp * settings.MINDAMPING) + settings.MAXDAMPING;
        points.push((
            Smooth::<Pos2> {
                value: Pos2 { x: 0., y: 0. },
                velocity: Pos2 { x: 0., y: 0. },
                target: Pos2 { x: 0., y: 0. },
                speed,
                damping,
            },
            color,
        ));
    }
    points
}

fn select_input_dev() -> cpal::Device {
    let mut devs = list_input_devs();
    assert!(!devs.is_empty(), "no input devices found!");
    if devs.len() == 1 {
        return devs.remove(0).1;
    }
    println!();
    devs.iter().enumerate().for_each(|(i, (name, dev))| {
        println!(
            "  [{}] {} {:?}",
            i,
            name,
            dev.default_input_config().unwrap()
        );
    });
    let mut input = String::new();

    stdin().lock().read_line(&mut input).unwrap();
    let index = input[0..1].parse::<usize>().unwrap();
    devs.remove(index).1
}

pub fn init_ringbuffer(sampling_rate: usize) -> Arc<Mutex<AllocRingBuffer<f32>>> {
    let mut buf: AllocRingBuffer<f32> =
        AllocRingBuffer::with_capacity((5 * sampling_rate).next_power_of_two());
    buf.fill(0.0);
    Arc::new(Mutex::new(buf))
}
