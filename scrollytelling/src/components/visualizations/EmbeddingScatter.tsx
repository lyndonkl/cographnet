import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface Point {
  x: number;
  y: number;
  category: 'diet' | 'nutrition';
}

const generateData = (): Point[] => {
  // Generate two overlapping clusters
  const data: Point[] = [];
  for (let i = 0; i < 50; i++) {
    // Diet cluster
    data.push({
      x: d3.randomNormal(0, 0.5)(),
      y: d3.randomNormal(0, 0.5)(),
      category: 'diet'
    });
    // Nutrition cluster (slightly offset)
    data.push({
      x: d3.randomNormal(0.3, 0.5)(),
      y: d3.randomNormal(0.3, 0.5)(),
      category: 'nutrition'
    });
  }
  return data;
};

export const EmbeddingScatter = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const data = generateData();

  useEffect(() => {
    if (!svgRef.current) return;

    // Clear previous
    d3.select(svgRef.current).selectAll("*").remove();

    // Setup
    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Scales
    const x = d3.scaleLinear()
      .domain(d3.extent(data, d => d.x) as [number, number])
      .range([0, innerWidth]);

    const y = d3.scaleLinear()
      .domain(d3.extent(data, d => d.y) as [number, number])
      .range([innerHeight, 0]);

    // SVG setup
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add points
    svg.selectAll('circle')
      .data(data)
      .enter()
      .append('circle')
      .attr('cx', d => x(d.x))
      .attr('cy', d => y(d.y))
      .attr('r', 5)
      .attr('class', d => 
        d.category === 'diet' 
          ? 'fill-indigo-400 opacity-60' 
          : 'fill-emerald-400 opacity-60'
      )
      .attr('stroke', 'white')
      .attr('stroke-width', 1);

    // Add legend
    const legend = svg.append('g')
      .attr('transform', `translate(${innerWidth - 100}, 0)`);

    ['diet', 'nutrition'].forEach((category, i) => {
      const g = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      g.append('circle')
        .attr('r', 5)
        .attr('class', category === 'diet' 
          ? 'fill-indigo-400' 
          : 'fill-emerald-400'
        );

      g.append('text')
        .attr('x', 10)
        .attr('y', 4)
        .text(category)
        .attr('class', 'text-sm fill-white');
    });

  }, [data]);

  return (
    <div className="w-full flex justify-center">
      <svg 
        ref={svgRef}
        className="max-w-full h-auto bg-slate-900/50 rounded-lg backdrop-blur-sm"
      />
    </div>
  );
}; 