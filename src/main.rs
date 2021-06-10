use std::{mem::swap, time::Instant};

use ::image::*;
use piston_window::*;
use rand::random;
use smore::*;

fn main() {
    // Load image
    let image = open("leaf.png").unwrap().to_rgb8();

    // Build areas
    let mut areas = Vec::new();
    let width = image.width() as i32;
    let height = image.height() as i32;
    let total_pixels = (width * height) as usize;
    let pixel = |i: i32, j: i32| {
        Cell(*image.get_pixel(((i + width) % width) as u32, ((j + height) % height) as u32))
    };
    const MAPPING_COUNT: usize = 10;
    for i in 0..width {
        for j in 0..height {
            if (i * height + j) as usize % (total_pixels / MAPPING_COUNT) != 0
                && (j * width + i) as usize % (total_pixels / MAPPING_COUNT) != 0
            {
                continue;
            }
            areas.push(Area([
                [pixel(i - 1, j - 1), pixel(i, j - 1), pixel(i + 1, j - 1)],
                [pixel(i - 1, j), pixel(i, j), pixel(i + 1, j)],
                [pixel(i - 1, j + 1), pixel(i, j + 1), pixel(i + 1, j + 1)],
            ]));
        }
    }

    dbg!(areas.len());

    // Train
    let mut model = Smore::new();
    for area in areas {
        model.map(&area, &area.0[1][1]);
    }

    // Create window
    const WINDOW_SIZE: f64 = 800.0;
    let mut window: PistonWindow = WindowSettings::new("Smore", [WINDOW_SIZE; 2])
        .build()
        .unwrap();

    // Init grids
    const SIZE: usize = 100;
    const SCALE: f64 = WINDOW_SIZE / SIZE as f64;
    let mut a = [[Cell(Rgb([0, 0, 0])); SIZE]; SIZE];
    for col in &mut a {
        for cell in col {
            cell.0[0] = random();
            cell.0[1] = random();
            cell.0[2] = random();
        }
    }
    let mut b = a;
    let curr = &mut a;
    let next = &mut b;

    let mut timer = Instant::now();

    let weight = Threshold::new(
        Exponential(10.0),
        &Area([[Cell(Rgb([255, 100, 0])); 3]; 3]),
        &Area([[Cell(Rgb([255, 255, 0])); 3]; 3]),
    );

    while let Some(event) = window.next() {
        window.draw_2d(&event, |Context { transform, .. }, graphics, _| {
            clear([0.0; 4], graphics);

            for (i, col) in curr.iter().enumerate() {
                for (j, cell) in col.iter().enumerate() {
                    rectangle(
                        [
                            cell.0[0] as f32 / 255.0,
                            cell.0[1] as f32 / 255.0,
                            cell.0[2] as f32 / 255.0,
                            1.0,
                        ],
                        [i as f64 * SCALE, j as f64 * SCALE, SCALE, SCALE],
                        transform,
                        graphics,
                    )
                }
            }
        });
        if let Event::Loop(Loop::Update(_)) = event {
            let now = Instant::now();
            if (now - timer).as_secs_f32() > 1.0 / 60.0 {
                timer = now;

                for i in 0..SIZE as i32 {
                    for j in 0..SIZE as i32 {
                        let curr_cell = |i: i32, j: i32| {
                            curr[((i + SIZE as i32) % SIZE as i32) as usize]
                                [((j + SIZE as i32) % SIZE as i32) as usize]
                        };
                        let curr_area = Area([
                            [
                                curr_cell(i - 1, j - 1),
                                curr_cell(i, j - 1),
                                curr_cell(i + 1, j - 1),
                            ],
                            [curr_cell(i - 1, j), curr_cell(i, j), curr_cell(i + 1, j)],
                            [
                                curr_cell(i - 1, j + 1),
                                curr_cell(i, j + 1),
                                curr_cell(i + 1, j + 1),
                            ],
                        ]);
                        let next_cell = &mut next[((i + SIZE as i32) % SIZE as i32) as usize]
                            [((j + SIZE as i32) % SIZE as i32) as usize];

                        *next_cell = model.eval(weight).get(&curr_area);
                    }
                }
                swap(curr, next);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Cell(pub Rgb<u8>);

const CELL_N: usize = 3;

impl Vectorize<CELL_N> for Cell {
    fn vectorize(&self) -> [f32; CELL_N] {
        [
            self.0[0] as f32 / 255.0,
            self.0[1] as f32 / 255.0,
            self.0[2] as f32 / 255.0,
        ]
    }
}

impl Devectorize<CELL_N> for Cell {
    fn devectorize(vector: [f32; CELL_N]) -> Self {
        Cell(Rgb([
            (vector[0].clamp(0.0, 1.0) * 255.0) as u8,
            (vector[1].clamp(0.0, 1.0) * 255.0) as u8,
            (vector[2].clamp(0.0, 1.0) * 255.0) as u8,
        ]))
    }
}

#[derive(Debug, Clone, Copy)]
struct Area(pub [[Cell; 3]; 3]);

const AREA_N: usize = 8 * CELL_N;

impl Vectorize<AREA_N> for Area {
    fn vectorize(&self) -> [f32; AREA_N] {
        let mut vector = [0.0; AREA_N];
        let mut c = 0;
        for i in 0..3 {
            for j in 0..3 {
                if i == 1 && j == 1 {
                    continue;
                }
                let pixel = self.0[i][j].0;

                for k in 0..3 {
                    vector[c + k] = pixel[k] as f32 / 255.0;
                }
                c += 3;
            }
        }
        vector
    }
}
